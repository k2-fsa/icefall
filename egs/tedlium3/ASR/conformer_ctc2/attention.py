# Copyright    2022  Behavox LLC.        (author: Daniil Kulko)
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

from typing import Optional, Tuple, Union

import torch
from scaling import ScaledLinear


class MultiheadAttention(torch.nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces. This is a modified
    version of the original version of multihead attention
    (see Attention Is All You Need <https://arxiv.org/abs/1706.03762>)
    with replacement of input / output projection layers
    with newly introduced ScaleLinear layer
    (see https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/scaling.py).

    Args:
        embed_dim:
          total dimension of the model.
        num_heads:
          number of parallel attention heads. Note that embed_dim will be split
          across num_heads, i.e. each head will have dimension (embed_dim // num_heads).
        dropout:
          dropout probability on attn_output_weights. (default=0.0).
        bias:
          if specified, adds bias to input / output projection layers (default=True).
        add_bias_kv:
          if specified, adds bias to the key and value sequences at dim=0 (default=False).
        add_zero_attn:
          if specified, adds a new batch of zeros to the key and value sequences
          at dim=1 (default=False).
        batch_first:
          if True, then the input and output tensors are provided as
          (batch, seq, feature), otherwise (seq, batch, feature) (default=False).

    Examples::
        >>> multihead_attn = MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        batch_first: bool = False,
        device: Union[torch.device, str, None] = None,
        dtype: Union[torch.dtype, str, None] = None,
    ) -> None:

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads. "
                "Got embedding dim vs number 0f heads: "
                f"{embed_dim} vs {num_heads}"
            )

        self.head_dim = embed_dim // num_heads

        self.in_proj = ScaledLinear(
            embed_dim,
            3 * embed_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.out_proj = ScaledLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            initial_scale=0.25,
            device=device,
            dtype=dtype,
        )

        if add_bias_kv:
            self.bias_k = torch.nn.Parameter(
                torch.empty((1, 1, embed_dim), device=device, dtype=dtype)
            )
            self.bias_v = torch.nn.Parameter(
                torch.empty((1, 1, embed_dim), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query:
              Query embeddings of shape (L, N, E_q) when batch_first=False or (N, L, E_q)
              when batch_first=True, where L is the target sequence length, N is the batch size,
              and E_q is the query embedding dimension embed_dim. Queries are compared against
              key-value pairs to produce the output. See "Attention Is All You Need" for more details.
            key:
              Key embeddings of shape (S, N, E_k) when batch_first=False or (N, S, E_k) when
              batch_first=True, where S is the source sequence length, N is the batch size, and
              E_k is the key embedding dimension kdim. See "Attention Is All You Need" for more details.
            value:
              Value embeddings of shape (S, N, E_v) when batch_first=False or (N, S, E_v) when
              batch_first=True, where S is the source sequence length, N is the batch size, and
              E_v is the value embedding dimension vdim. See "Attention Is All You Need" for more details.
            key_padding_mask:
              If specified, a mask of shape (N, S) indicating which elements within key
              to ignore for the purpose of attention (i.e. treat as "padding").
              Binary and byte masks are supported. For a binary mask, a True value indicates
              that the corresponding key value will be ignored for the purpose of attention.
              For a byte mask, a non-zero value indicates that the corresponding key value will be ignored.
            need_weights:
              If specifid, returns attn_output_weights in addition to attn_outputs (default=True).
            attn_mask:
              If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
              (L, S) or (N * num_heads, L, S), where N is the batch size, L is the target sequence length,
              and S is the source sequence length. A 2D mask will be broadcasted across the batch while
              a 3D mask allows for a different mask for each entry in the batch.
              Binary, byte, and float masks are supported. For a binary mask, a True value indicates
              that the corresponding position is not allowed to attend. For a byte mask, a non-zero
              value indicates that the corresponding position is not allowed to attend. For a float mask,
              the mask values will be added to the attention weight.

        Returns:
            attn_output:
              Attention outputs of shape (L, N, E) when batch_first=False or (N, L, E) when batch_first=True,
              where L is the target sequence length, N is the batch size, and E is the embedding dimension
              embed_dim.
            attn_output_weights:
              Attention output weights of shape (N, L, S), where N is the batch size, L is the target sequence
              length, and S is the source sequence length. Only returned when need_weights=True.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        (
            attn_output,
            attn_output_weights,
        ) = torch.nn.functional.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight=self.in_proj.get_weight(),
            in_proj_bias=self.in_proj.get_bias(),
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.get_weight(),
            out_proj_bias=self.out_proj.get_bias(),
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        return attn_output, attn_output_weights
