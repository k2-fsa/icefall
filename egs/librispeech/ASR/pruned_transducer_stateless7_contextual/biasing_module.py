# Copyright    2024  (authors: Ruizhe Huang)
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

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Ffn(nn.Module):
    """A stack of Linear -> Tanh -> Dropout layers, optionally with a
    residual connection around the whole stack."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        nlayers: int = 1,
        drop_out: float = 0.1,
        skip: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        for ilayer in range(nlayers):
            _in = hidden_dim if ilayer > 0 else input_dim
            _out = hidden_dim if ilayer < nlayers - 1 else out_dim
            layers.extend(
                [
                    nn.Linear(_in, _out),
                    nn.Tanh(),
                    nn.Dropout(p=drop_out),
                ]
            )
        self.ffn = torch.nn.Sequential(*layers)

        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.ffn(x)

        if self.skip:
            x_out = x_out + x

        return x_out


class BiasingModule(torch.nn.Module):
    """Cross-attention from the encoder (or decoder) frames to the embeddings
    of the biasing words. Its output is added to the encoder (or decoder)
    output by the caller."""

    def __init__(
        self,
        query_dim: int,
        qkv_dim: int = 64,
        num_heads: int = 4,
    ):
        """
        Args:
          query_dim:
            Dimension of the encoder (or decoder) output.
          qkv_dim:
            Dimension of the attention; it must equal the dimension of the
            context embeddings.
          num_heads:
            Number of attention heads.
        """
        super().__init__()
        self.proj_in1 = nn.Linear(query_dim, qkv_dim)
        self.proj_in2 = Ffn(
            input_dim=qkv_dim,
            hidden_dim=qkv_dim,
            out_dim=qkv_dim,
            skip=True,
            drop_out=0.1,
            nlayers=2,
        )
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=qkv_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj_out1 = Ffn(
            input_dim=qkv_dim,
            hidden_dim=qkv_dim,
            out_dim=qkv_dim,
            skip=True,
            drop_out=0.1,
            nlayers=2,
        )
        self.proj_out2 = nn.Linear(qkv_dim, query_dim)
        self.glu = nn.GLU()

    def forward(
        self,
        queries: torch.Tensor,
        contexts: torch.Tensor,
        contexts_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          queries:
            A tensor of shape (batch_size, seq_length, query_dim).
          contexts:
            A tensor of shape (batch_size, max_contexts_size, qkv_dim).
          contexts_mask:
            A bool tensor of shape (batch_size, max_contexts_size); True
            marks padding positions.
          need_weights:
            If True, also return the attention weights.
        Returns:
          A tuple (biasing_output, attn_weights), where biasing_output has
          shape (batch_size, seq_length, query_dim) and attn_weights has shape
          (batch_size, seq_length, max_contexts_size), or is None.
        """
        _queries = self.proj_in1(queries)
        _queries = self.proj_in2(_queries)

        attn_output, attn_output_weights = self.multihead_attn(
            _queries,  # query
            contexts,  # key
            contexts,  # value
            key_padding_mask=contexts_mask,
            need_weights=need_weights,
        )
        output = self.proj_out1(attn_output)
        output = self.proj_out2(output)

        # GLU on [output, output], i.e., output * sigmoid(output)
        biasing_output = self.glu(output.repeat(1, 1, 2))

        return biasing_output, attn_output_weights
