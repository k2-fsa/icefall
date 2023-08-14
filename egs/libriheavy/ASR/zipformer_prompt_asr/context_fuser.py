# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from scaling import ScaledLinear, softmax

from icefall.utils import make_pad_mask

class ContextFuser(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
    
    def forward(
        self,
        context: torch.Tensor,
        context_lens: torch.Tensor=None,
        padding_mask: torch.Tensor=None
    ) -> torch.Tensor:
        """A module fusing the context embedding vectors

        Args:
            context (torch.Tensor): The context embeddings, (B,W,C)
            context_lens (torch.Tensor): The length of context embeddings, (B,)

        Returns:
            torch.Tensor: The fused context embeddings, (B,C)
        """
        batch_size = context.size(0)
        if padding_mask is None:
            assert context_lens is not None
            padding_mask = make_pad_mask(context_lens).unsqueeze(-1)
        else:
            if padding_mask.ndim != 3:
                padding_mask = padding_mask.unsqueeze(-1)
        context.masked_fill_(padding_mask, 0)
        
        if context_lens is None:
            max_len = padding_mask.size(1)
            context_lens = max_len - padding_mask.sum(dim=1) + 1e-5 # to prevent 0
        
        # by a small probability, dropout the context of a few samples
        context_dropout_rate = 0.05
        m = torch.rand((batch_size, 1, 1), device=context.device) > context_dropout_rate
        context = context * m 
        
        # average the context
        context = context.sum(dim=1)/context_lens
        
        return context
    
class SelfAttContextFuser(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        query_head_dim: int = 128,
        nhead: int=4,
        context_dropout_rate: float=0.05,
    ):
        """ContextFuser with multi-head self-attention

        Args:
            embed_dim (int, optional): The input embedding dim. Defaults to 256.
            nhead (int, optional): The number of heads. Defaults to 4.
        """
        
        super().__init__()
        
        self.embed_dim = embed_dim 
        self.nhead = nhead
        self.query_head_dim = query_head_dim
        
        self.in_proj = ScaledLinear(embed_dim, nhead * query_head_dim)
        self.weight_proj = ScaledLinear(nhead * query_head_dim, nhead)
        self.context_dropout_rate = context_dropout_rate
    
    def forward(
        self,
        context: torch.Tensor,
        context_lens: torch.Tensor=None,
        padding_mask: torch.Tensor=None,
    ) -> torch.Tensor:
        """A module fusing the context embedding vectors

        Args:
            context (torch.Tensor): The context embeddings, (B,W,C)
            context_lens (torch.Tensor): The length of context embeddings, (B,)
            padding_mask (torch.Tensor): A padding mask (B,W)

        Returns:
            torch.Tensor: The fused context embeddings, (B,C)
        """
        batch_size = context.size(0)
        if padding_mask is None:
            assert context_lens is not None
            padding_mask = make_pad_mask(context_lens).unsqueeze(-1)
        else:
            if padding_mask.ndim != 3:
                padding_mask = padding_mask.unsqueeze(-1)
        # context.masked_fill_(padding_mask, 0) 
        
        if context_lens is None:
            max_len = padding_mask.size(1)
            context_lens = max_len - padding_mask.sum(dim=1) + 1e-5 # to prevent 0
            
        k = self.in_proj(context) # (B,W,C)
        w = self.weight_proj(torch.tanh(k)) # (B,W,num_heads)
        
        w.masked_fill_(padding_mask, -1000)
        w = softmax(w, dim=1) # (B,W,num_heads)
        w = w.permute(0,2,1) # (B,num_heads, W)
        
        # reweight and concat the context embeddings
        context = torch.matmul(w, context).view(batch_size, -1) # (B, num_heads * C)
        
        # by a small probability, dropout the context of a few samples
        if self.training:
            m = torch.rand((batch_size, 1), device=context.device) > self.context_dropout_rate
            context = context * m
        #context = context * 0.0
        
        return context