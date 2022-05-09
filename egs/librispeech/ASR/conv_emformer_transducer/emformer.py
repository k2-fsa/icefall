# Copyright      2022  Xiaomi Corporation     (Author: Zengwei Yao)
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
#
# It is modified based on https://github.com/pytorch/audio/blob/main/torchaudio/models/emformer.py.  # noqa

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from subsampling import Conv2dSubsampling, VggSubsampling

from icefall.utils import make_pad_mask


def _gen_attention_mask_block(
    col_widths: List[int],
    col_mask: List[bool],
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    assert len(col_widths) == len(
        col_mask
    ), "Length of col_widths must match that of col_mask"

    mask_block = [
        torch.ones(num_rows, col_width, device=device)
        if is_ones_col
        else torch.zeros(num_rows, col_width, device=device)
        for col_width, is_ones_col in zip(col_widths, col_mask)
    ]
    return torch.cat(mask_block, dim=1)


class EmformerAttention(nn.Module):
    r"""Emformer layer attention module.

    Args:
      embed_dim (int):
        Embedding dimension.
      nhead (int):
        Number of attention heads in each Emformer layer.
      tanh_on_mem (bool, optional):
        If ``True``, applies tanh to memory elements. (Default: ``False``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if embed_dim % nhead != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) is not a multiple of"
                f"nhead ({nhead})."
            )

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf
        self.head_dim = embed_dim // nhead

        self.scaling = self.head_dim ** -0.5

        self.emb_to_key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.emb_to_query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3  # noqa
        self.pos_bias_u = nn.Parameter(torch.Tensor(nhead, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(nhead, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb_to_key_value.weight)
        nn.init.constant_(self.emb_to_key_value.bias, 0.0)

        nn.init.xavier_uniform_(self.emb_to_query.weight)
        nn.init.constant_(self.emb_to_query.bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_pos.weight)

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def _gen_attention_probs(
        self,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Given the entire attention weights, mask out unecessary connections
        and optionally with padding positions, to obtain underlying chunk-wise
        attention probabilities.

        B: batch size;
        Q: length of query;
        KV: length of key and value.

        Args:
          attention_weights (torch.Tensor):
            Attention weights computed on the entire concatenated tensor
            with shape (B * nhead, Q, KV).
          attention_mask (torch.Tensor):
            Mask tensor where chunk-wise connections are filled with `False`,
            and other unnecessary connections are filled with `True`,
            with shape (Q, KV).
          padding_mask (torch.Tensor, optional):
            Mask tensor where the padding positions are fill with `True`,
            and other positions are filled with `False`, with shapa `(B, KV)`.

        Returns:
          A tensor of shape (B * nhead, Q, KV).
        """
        attention_weights_float = attention_weights.float()
        attention_weights_float = attention_weights_float.masked_fill(
            attention_mask.unsqueeze(0), self.negative_inf
        )
        if padding_mask is not None:
            Q = attention_weights.size(1)
            B = attention_weights.size(0) // self.nhead
            attention_weights_float = attention_weights_float.view(
                B, self.nhead, Q, -1
            )
            attention_weights_float = attention_weights_float.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                self.negative_inf,
            )
            attention_weights_float = attention_weights_float.view(
                B * self.nhead, Q, -1
            )

        attention_probs = nn.functional.softmax(
            attention_weights_float, dim=-1
        ).type_as(attention_weights)

        return attention_probs

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
          x: Input tensor, of shape (B, nhead, U, PE).
             U is the length of query vector.
             For non-infer mode, PE = 2 * U - 1;
             for infer mode, PE = L + 2 * U - 1.

        Returns:
          A tensor of shape (B, nhead, U, out_len).
          For non-infer mode, out_len = U;
          for infer mode, out_len = L + U.
        """
        B, nhead, U, PE = x.size()
        B_stride = x.stride(0)
        nhead_stride = x.stride(1)
        U_stride = x.stride(2)
        PE_stride = x.stride(3)
        out_len = PE - (U - 1)
        return x.as_strided(
            size=(B, nhead, U, out_len),
            stride=(B_stride, nhead_stride, U_stride - PE_stride, PE_stride),
            storage_offset=PE_stride * (U - 1),
        )

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Underlying chunk-wise attention implementation.

        L: length of left_context;
        S: length of summary;
        M: length of memory;
        Q: length of attention query;
        KV: length of attention key and value.

        1) Concat right_context, utterance, summary,
        and compute query tensor with length Q = R + U + S.
        2) Concat memory, right_context, utterance,
        and compute key, value tensors with length KV = M + R + U;
        optionally with left_context_key and left_context_val (inference mode),
        then KV = M + R + L + U.
        3) Compute entire attention scores with query, key, and value,
        then apply attention_mask to get underlying chunk-wise attention scores.

        Args:
          utterance (torch.Tensor):
            Utterance frames, with shape (U, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing
            number of valid frames for i-th batch element in utterance.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D).
          summary (torch.Tensor):
            Summary elements, with shape (S, B, D).
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D).
          attention_mask (torch.Tensor):
            Attention mask for underlying attention, with shape (Q, KV).
          pos_emb (torch.Tensor):
            Position encoding embedding, with shape (PE, D).
            For training mode, PE = 2*U-1;
            For infer mode, PE = L+2*U-1.
          left_context_key (torch,Tensor, optional):
            Cached attention key of left context from preceding computation,
            with shape (L, B, D).
          left_context_val (torch.Tensor, optional):
            Cached attention value of left context from preceding computation,
            with shape (L, B, D).

        Returns:
          A tuple containing 4 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (S, B, D).
            - attention key, with shape (KV, B, D).
            - attention value, with shape (KV, B, D).
        """
        U, B, _ = utterance.size()
        R = right_context.size(0)
        M = memory.size(0)

        # Compute query with [right context, utterance, summary].
        query = self.emb_to_query(
            torch.cat([right_context, utterance, summary])
        )
        # Compute key and value with [mems, right context, utterance].
        key, value = self.emb_to_key_value(
            torch.cat([memory, right_context, utterance])
        ).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            # This is for inference mode. Now compute key and value with
            # [mems, right context, left context, uttrance]
            key = torch.cat(
                [key[: M + R], left_context_key, key[M + R :]]  # noqa
            )
            value = torch.cat(
                [value[: M + R], left_context_val, value[M + R :]]  # noqa
            )
        Q = query.size(0)
        KV = key.size(0)

        reshaped_key, reshaped_value = [
            tensor.contiguous()
            .view(KV, B * self.nhead, self.head_dim)
            .transpose(0, 1)
            for tensor in [key, value]
        ]  # (B * nhead, KV, head_dim)
        reshaped_query = query.contiguous().view(
            Q, B, self.nhead, self.head_dim
        )

        # compute attention matrix ac
        query_with_bais_u = (
            (reshaped_query + self.pos_bias_u)
            .view(Q, B * self.nhead, self.head_dim)
            .transpose(0, 1)
        )
        matrix_ac = torch.bmm(
            query_with_bais_u, reshaped_key.transpose(1, 2)
        )  # (B * nhead, Q, KV)

        # compute attention matrix bd
        utterance_with_bais_v = (
            reshaped_query[R : R + U] + self.pos_bias_v
        ).permute(1, 2, 0, 3)
        # (B, nhead, U, head_dim)
        PE = pos_emb.size(0)
        if left_context_key is not None and left_context_val is not None:
            L = left_context_key.size(0)
            assert PE == L + 2 * U - 1
        else:
            assert PE == 2 * U - 1
        pos_emb = (
            self.linear_pos(pos_emb)
            .view(PE, self.nhead, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )  # (1, nhead, PE, head_dim)
        matrix_bd_utterance = torch.matmul(
            utterance_with_bais_v, pos_emb.transpose(-2, -1)
        )  # (B, nhead, U, PE)
        # rel-shift
        matrix_bd_utterance = self._rel_shift(
            matrix_bd_utterance
        )  # (B, nhead, U, U or L + U)
        matrix_bd_utterance = matrix_bd_utterance.contiguous().view(
            B * self.nhead, U, -1
        )
        matrix_bd = torch.zeros_like(matrix_ac)
        matrix_bd[:, R : R + U, M + R :] = matrix_bd_utterance

        attention_weights = (matrix_ac + matrix_bd) * self.scaling

        # Compute padding mask
        if B == 1:
            padding_mask = None
        else:
            padding_mask = make_pad_mask(KV - U + lengths)

        # Compute attention probabilities.
        attention_probs = self._gen_attention_probs(
            attention_weights, attention_mask, padding_mask
        )

        # Compute attention.
        attention = torch.bmm(attention_probs, reshaped_value)
        assert attention.shape == (B * self.nhead, Q, self.head_dim)
        attention = (
            attention.transpose(0, 1).contiguous().view(Q, B, self.embed_dim)
        )

        # Apply output projection.
        outputs = self.out_proj(attention)

        output_right_context_utterance = outputs[: R + U]
        output_memory = outputs[R + U :]
        if self.tanh_on_mem:
            output_memory = torch.tanh(output_memory)
        else:
            output_memory = torch.clamp(output_memory, min=-10, max=10)

        return output_right_context_utterance, output_memory, key, value

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Modify docs.
        """Forward pass for training.

        B: batch size;
        D: embedding dimension;
        R: length of right_context;
        U: length of utterance;
        S: length of summary;
        M: length of memory.

        Args:
          utterance (torch.Tensor):
            Utterance frames, with shape (U, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing
            number of valid frames for i-th batch element in utterance.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D).
          summary (torch.Tensor):
            Summary elements, with shape (S, B, D).
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D).
          attention_mask (torch.Tensor):
            Attention mask for underlying chunk-wise attention,
            with shape (Q, KV), where Q = R + U + S, KV = M + R + U.
          pos_emb (torch.Tensor):
            Position encoding embedding, with shape (PE, D).
            For training mode, P = 2*U-1.

        Returns:
          A tuple containing 2 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (M, B, D), where M = S - 1 or M = 0.
        """
        (
            output_right_context_utterance,
            output_memory,
            _,
            _,
        ) = self._forward_impl(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            attention_mask,
            pos_emb,
        )
        return output_right_context_utterance, output_memory[:-1]

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        left_context_key: torch.Tensor,
        left_context_val: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        B: batch size;
        D: embedding dimension;
        R: length of right_context;
        U: length of utterance;
        L: length of left_context;
        S: length of summary;
        M: length of memory;

        Args:
          utterance (torch.Tensor):
            Utterance frames, with shape (U, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing
            number of valid frames for i-th batch element in utterance.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D).
          summary (torch.Tensor):
            Summary element, with shape (1, B, D), or empty.
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D).
          left_context_key (torch,Tensor):
            Cached attention key of left context from preceding computation,
            with shape (L, B, D).
          left_context_val (torch.Tensor):
            Cached attention value of left context from preceding computation,
            with shape (L, B, D).
          pos_emb (torch.Tensor):
            Position encoding embedding, with shape (PE, D).
            For infer mode, PE = L+2*U-1.

        Returns:
          A tuple containing 4 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (1, B, D) or (0, B, D).
            - attention key of left context and utterance, which would be cached
              for next computation, with shape (L + U, B, D).
            - attention value of left context and utterance, which would be
              cached for next computation, with shape (L + U, B, D).
        """
        # query: [right context, utterance, summary]
        Q = right_context.size(0) + utterance.size(0) + summary.size(0)
        # key, value: [memory, right context, left context, uttrance]
        KV = (
            memory.size(0)
            + right_context.size(0)  # noqa
            + left_context_key.size(0)  # noqa
            + utterance.size(0)  # noqa
        )
        attention_mask = torch.zeros(Q, KV).to(
            dtype=torch.bool, device=utterance.device
        )
        # Disallow attention bettween the summary vector with the memory bank
        attention_mask[-1, : memory.size(0)] = True
        (
            output_right_context_utterance,
            output_memory,
            key,
            value,
        ) = self._forward_impl(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            attention_mask,
            pos_emb,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
        )
        right_context_end_idx = memory.size(0) + right_context.size(0)
        return (
            output_right_context_utterance,
            output_memory,
            key[right_context_end_idx:],
            value[right_context_end_idx:],
        )


class EmformerLayer(nn.Module):
    """Emformer layer that constitutes Emformer.

    Args:
      d_model (int):
        Input dimension.
      nhead (int):
        Number of attention heads.
      dim_feedforward (int):
        Hidden layer dimension of feedforward network.
      chunk_length (int):
        Length of each input segment.
      dropout (float, optional):
        Dropout probability. (Default: 0.0)
      cnn_module_kernel (int):
        Kernel size of convolution module.
      left_context_length (int, optional):
        Length of left context. (Default: 0)
      max_memory_size (int, optional):
        Maximum number of memory elements to use. (Default: 0)
      tanh_on_mem (bool, optional):
        If ``True``, applies tanh to memory elements. (Default: ``False``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (Default: -1e8)
      causal (bool):
        Whether use causal convolution (default=False).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        chunk_length: int,
        dropout: float = 0.0,
        cnn_module_kernel: int = 3,
        left_context_length: int = 0,
        max_memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        causal: bool = False,
    ):
        super().__init__()

        self.attention = EmformerAttention(
            embed_dim=d_model,
            nhead=nhead,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )
        self.summary_op = nn.AvgPool1d(
            kernel_size=chunk_length, stride=chunk_length, ceil_mode=True
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = ConvolutionModule(
            d_model,
            cnn_module_kernel,
            causal=causal,
        )

        self.norm_ff_macaron = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.norm_mha = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ff_scale = 0.5
        self.left_context_length = left_context_length
        self.chunk_length = chunk_length
        self.max_memory_size = max_memory_size
        self.d_model = d_model
        self.use_memory = max_memory_size > 0

    def _init_state(
        self, batch_size: int, device: Optional[torch.device]
    ) -> List[torch.Tensor]:
        """Initialize states with zeros."""
        empty_memory = torch.zeros(
            self.max_memory_size, batch_size, self.d_model, device=device
        )
        left_context_key = torch.zeros(
            self.left_context_length, batch_size, self.d_model, device=device
        )
        left_context_val = torch.zeros(
            self.left_context_length, batch_size, self.d_model, device=device
        )
        past_length = torch.zeros(
            1, batch_size, dtype=torch.int32, device=device
        )
        return [empty_memory, left_context_key, left_context_val, past_length]

    def _unpack_state(
        self, state: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack cached states including:
        1) output memory from previous chunks in the lower layer;
        2) attention key and value of left context from proceeding chunk's
        computation.
        """
        past_length = state[3][0][0].item()
        past_left_context_length = min(self.left_context_length, past_length)
        past_memory_length = min(
            self.max_memory_size, math.ceil(past_length / self.chunk_length)
        )
        memory_start_idx = self.max_memory_size - past_memory_length
        pre_memory = state[0][memory_start_idx:]
        left_context_start_idx = (
            self.left_context_length - past_left_context_length
        )
        left_context_key = state[1][left_context_start_idx:]
        left_context_val = state[2][left_context_start_idx:]
        return pre_memory, left_context_key, left_context_val

    def _pack_state(
        self,
        next_key: torch.Tensor,
        next_val: torch.Tensor,
        update_length: int,
        memory: torch.Tensor,
        state: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Pack updated states including:
        1) output memory of current chunk in the lower layer;
        2) attention key and value in current chunk's computation, which would
        be resued in next chunk's computation.
        3) length of current chunk.
        """
        new_memory = torch.cat([state[0], memory])
        new_key = torch.cat([state[1], next_key])
        new_val = torch.cat([state[2], next_val])
        memory_start_idx = new_memory.size(0) - self.max_memory_size
        state[0] = new_memory[memory_start_idx:]
        key_start_idx = new_key.size(0) - self.left_context_length
        state[1] = new_key[key_start_idx:]
        val_start_idx = new_val.size(0) - self.left_context_length
        state[2] = new_val[val_start_idx:]
        state[3] = state[3] + update_length
        return state

    def _apply_macaron_feed_foward_module(
        self, right_context_utterance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply macaron style feed forward module."""
        residual = right_context_utterance
        right_context_utterance = self.norm_ff_macaron(right_context_utterance)
        right_context_utterance = residual + self.ff_scale * self.dropout(
            self.feed_forward_macaron(right_context_utterance)
        )
        return right_context_utterance

    def _apply_feed_forward_module(
        self, right_context_utterance: torch.Tensor
    ) -> torch.Tensor:
        """Apply feed forward module."""
        residual = right_context_utterance
        right_context_utterance = self.norm_ff(right_context_utterance)
        right_context_utterance = residual + self.ff_scale * self.dropout(
            self.feed_forward(right_context_utterance)
        )
        return right_context_utterance

    def _apply_conv_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        right_context_end_idx: int,
    ) -> torch.Tensor:
        """Apply convolution module on utterance in non-infer mode."""
        utterance = right_context_utterance[right_context_end_idx:]
        right_context = right_context_utterance[:right_context_end_idx]

        residual = utterance
        utterance = self.norm_conv(utterance)
        utterance, _ = self.conv_module(utterance)
        utterance = residual + self.dropout(utterance)
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance

    def _apply_conv_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        right_context_end_idx: int,
        conv_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply convolution module on utterance in infer mode."""
        utterance = right_context_utterance[right_context_end_idx:]
        right_context = right_context_utterance[:right_context_end_idx]

        residual = utterance
        utterance = self.norm_conv(utterance)
        utterance, conv_cache = self.conv_module(utterance, conv_cache)
        utterance = residual + self.dropout(utterance)
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance, conv_cache

    def _apply_attention_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        right_context_end_idx: int,
        lengths: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention module in non-infer mode."""
        if attention_mask is None:
            raise ValueError(
                "attention_mask must be not None in non-infer mode. "
            )

        residual = right_context_utterance
        right_context_utterance = self.norm_mha(right_context_utterance)
        utterance = right_context_utterance[right_context_end_idx:]
        right_context = right_context_utterance[:right_context_end_idx]

        if self.use_memory:
            summary = self.summary_op(utterance.permute(1, 2, 0)).permute(
                2, 0, 1
            )
        else:
            summary = torch.empty(0).to(
                dtype=utterance.dtype, device=utterance.device
            )
        output_right_context_utterance, output_memory = self.attention(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            memory=memory,
            attention_mask=attention_mask,
            pos_emb=pos_emb,
        )
        right_context_utterance = residual + self.dropout(
            output_right_context_utterance
        )

        return right_context_utterance, output_memory

    def _apply_attention_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        right_context_end_idx: int,
        lengths: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Apply attention in infer mode.
        1) Unpack cached states including:
          - memory from previous chunks in the lower layer;
          - attention key and value of left context from proceeding
            chunk's compuation;
        2) Apply attention computation;
        3) Pack updated states including:
          - output memory of current chunk in the lower layer;
          - attention key and value in current chunk's computation, which would
            be resued in next chunk's computation.
          - length of current chunk.
        """
        residual = right_context_utterance
        right_context_utterance = self.norm_mha(right_context_utterance)
        utterance = right_context_utterance[right_context_end_idx:]
        right_context = right_context_utterance[:right_context_end_idx]

        if state is None:
            state = self._init_state(utterance.size(1), device=utterance.device)
        pre_memory, left_context_key, left_context_val = self._unpack_state(
            state
        )
        if self.use_memory:
            summary = self.summary_op(utterance.permute(1, 2, 0)).permute(
                2, 0, 1
            )
            summary = summary[:1]
        else:
            summary = torch.empty(0).to(
                dtype=utterance.dtype, device=utterance.device
            )
        # pos_emb is of shape [PE, D], PE = L + 2 * U - 1,
        # the relative distance j - i of key(j) and query(i) is in range of [-(L + U - 1), (U - 1)]  # noqa
        L = left_context_key.size(0)  # L <= left_context_length
        U = utterance.size(0)
        PE = L + 2 * U - 1
        tot_PE = self.left_context_length + 2 * U - 1
        assert pos_emb.size(0) == tot_PE
        pos_emb = pos_emb[tot_PE - PE :]
        (
            output_right_context_utterance,
            output_memory,
            next_key,
            next_val,
        ) = self.attention.infer(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            memory=pre_memory,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
            pos_emb=pos_emb,
        )
        right_context_utterance = residual + self.dropout(
            output_right_context_utterance
        )
        state = self._pack_state(
            next_key, next_val, utterance.size(0), memory, state
        )
        return right_context_utterance, output_memory, state

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.
        1) Apply layer normalization on input utterance and right context
        before attention;
        2) Apply attention module, compute updated utterance, right context,
        and memory;
        3) Apply feed forward module and layer normalization on output utterance
        and right context.

        B: batch size;
        D: embedding dimension;
        R: length of right_context;
        U: length of utterance;
        M: length of memory.

        Args:
          utterance (torch.Tensor):
            Utterance frames, with shape (U, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing
            number of valid frames for i-th batch element in utterance.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D).
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D).
          attention_mask (torch.Tensor):
            Attention mask for underlying attention module,
            with shape (Q, KV), where Q = R + U + S, KV = M + R + U.
          pos_emb (torch.Tensor):
            Position encoding embedding, with shape (PE, D).
            For training mode, P = 2*U-1.

        Returns:
          A tuple containing 3 tensors:
            - output utterance, with shape (U, B, D).
            - output right context, with shape (R, B, D).
            - output memory, with shape (M, B, D).
        """
        right_context_utterance = torch.cat([right_context, utterance])
        right_context_end_idx = right_context.size(0)

        right_context_utterance = self._apply_macaron_feed_foward_module(
            right_context_utterance
        )

        (
            right_context_utterance,
            output_memory,
        ) = self._apply_attention_module_forward(
            right_context_utterance,
            right_context_end_idx,
            lengths,
            memory,
            attention_mask,
            pos_emb,
        )

        right_context_utterance = self._apply_conv_module_forward(
            right_context_utterance, right_context_end_idx
        )

        right_context_utterance = self._apply_feed_forward_module(
            right_context_utterance
        )

        right_context_utterance = self.norm_final(right_context_utterance)

        output_utterance = right_context_utterance[right_context_end_idx:]
        output_right_context = right_context_utterance[:right_context_end_idx]
        return output_utterance, output_right_context, output_memory

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        conv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass for inference.

         1) Apply layer normalization on input utterance and right context
         before attention;
         2) Apply attention module with cached state, compute updated utterance,
         right context, and memory, and update state;
         3) Apply feed forward module and layer normalization on output
         utterance and right context.

         B: batch size;
         D: embedding dimension;
         R: length of right_context;
         U: length of utterance;
         M: length of memory.

        Args:
           utterance (torch.Tensor):
             Utterance frames, with shape (U, B, D).
           lengths (torch.Tensor):
             With shape (B,) and i-th element representing
             number of valid frames for i-th batch element in utterance.
           right_context (torch.Tensor):
             Right context frames, with shape (R, B, D).
           memory (torch.Tensor):
             Memory elements, with shape (M, B, D).
           state (List[torch.Tensor], optional):
             List of tensors representing layer internal state generated in
             preceding computation. (default=None)
           pos_emb (torch.Tensor):
             Position encoding embedding, with shape (PE, D).
             For infer mode, PE = L+2*U-1.
           conv_cache (torch.Tensor, optional):
             Cache tensor of left context for causal convolution.

         Returns:
           (Tensor, Tensor, List[torch.Tensor], Tensor):
             - output utterance, with shape (U, B, D);
             - output right_context, with shape (R, B, D);
             - output memory, with shape (1, B, D) or (0, B, D).
             - output state.
             - updated conv_cache.
        """
        right_context_utterance = torch.cat([right_context, utterance])
        right_context_end_idx = right_context.size(0)

        right_context_utterance = self._apply_macaron_feed_foward_module(
            right_context_utterance
        )

        (
            right_context_utterance,
            output_memory,
            output_state,
        ) = self._apply_attention_module_infer(
            right_context_utterance,
            right_context_end_idx,
            lengths,
            memory,
            pos_emb,
            state,
        )

        right_context_utterance, conv_cache = self._apply_conv_module_infer(
            right_context_utterance,
            right_context_end_idx,
            conv_cache,
        )

        right_context_utterance = self._apply_feed_forward_module(
            right_context_utterance
        )

        right_context_utterance = self.norm_final(right_context_utterance)

        output_utterance = right_context_utterance[right_context_end_idx:]
        output_right_context = right_context_utterance[:right_context_end_idx]
        return (
            output_utterance,
            output_right_context,
            output_memory,
            output_state,
            conv_cache,
        )


class EmformerEncoder(nn.Module):
    """Implements the Emformer architecture introduced in
    *Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency
    Streaming Speech Recognition*
    [:footcite:`shi2021emformer`].

    Args:
      d_model (int):
        Input dimension.
      nhead (int):
        Number of attention heads in each emformer layer.
      dim_feedforward (int):
        Hidden layer dimension of each emformer layer's feedforward network.
      num_encoder_layers (int):
        Number of emformer layers to instantiate.
      chunk_length (int):
        Length of each input segment.
      dropout (float, optional):
        Dropout probability. (default: 0.0)
      left_context_length (int, optional):
        Length of left context. (default: 0)
      right_context_length (int, optional):
        Length of right context. (default: 0)
      max_memory_size (int, optional):
        Maximum number of memory elements to use. (default: 0)
      tanh_on_mem (bool, optional):
        If ``true``, applies tanh to memory elements. (default: ``false``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (default: -1e8)
      causal (bool):
        Whether use causal convolution (default=False).
    """

    def __init__(
        self,
        chunk_length: int,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        cnn_module_kernel: int = 3,
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        causal: bool = False,
    ):
        super().__init__()

        self.use_memory = max_memory_size > 0
        self.init_memory_op = nn.AvgPool1d(
            kernel_size=chunk_length,
            stride=chunk_length,
            ceil_mode=True,
        )

        self.emformer_layers = nn.ModuleList(
            [
                EmformerLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    chunk_length,
                    dropout=dropout,
                    cnn_module_kernel=cnn_module_kernel,
                    left_context_length=left_context_length,
                    max_memory_size=max_memory_size,
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                    causal=causal,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        self.max_memory_size = max_memory_size

    def _gen_right_context(self, x: torch.Tensor) -> torch.Tensor:
        """Hard copy each chunk's right context and concat them."""
        T = x.shape[0]
        num_segs = math.ceil(
            (T - self.right_context_length) / self.chunk_length
        )
        right_context_blocks = []
        for seg_idx in range(num_segs - 1):
            start = (seg_idx + 1) * self.chunk_length
            end = start + self.right_context_length
            right_context_blocks.append(x[start:end])
        last_right_context_start_idx = T - self.right_context_length
        right_context_blocks.append(x[last_right_context_start_idx:])
        return torch.cat(right_context_blocks)

    def _gen_attention_mask_col_widths(
        self, chunk_idx: int, U: int
    ) -> List[int]:
        """Calculate column widths (key, value) in attention mask for the
        chunk_idx chunk."""
        num_chunks = math.ceil(U / self.chunk_length)
        rc = self.right_context_length
        lc = self.left_context_length
        rc_start = chunk_idx * rc
        rc_end = rc_start + rc
        chunk_start = max(chunk_idx * self.chunk_length - lc, 0)
        chunk_end = min((chunk_idx + 1) * self.chunk_length, U)
        R = rc * num_chunks

        if self.use_memory:
            m_start = max(chunk_idx - self.max_memory_size, 0)
            M = num_chunks - 1
            col_widths = [
                m_start,  # before memory
                chunk_idx - m_start,  # memory
                M - chunk_idx,  # after memory
                rc_start,  # before right context
                rc,  # right context
                R - rc_end,  # after right context
                chunk_start,  # before chunk
                chunk_end - chunk_start,  # chunk
                U - chunk_end,  # after chunk
            ]
        else:
            col_widths = [
                rc_start,  # before right context
                rc,  # right context
                R - rc_end,  # after right context
                chunk_start,  # before chunk
                chunk_end - chunk_start,  # chunk
                U - chunk_end,  # after chunk
            ]

        return col_widths

    def _gen_attention_mask(self, utterance: torch.Tensor) -> torch.Tensor:
        """Generate attention mask for underlying chunk-wise attention
        computation, where chunk-wise connections are filled with `False`,
        and other unnecessary connections beyond chunk are filled with `True`.

        R: length of right_context;
        U: length of utterance;
        S: length of summary;
        M: length of memory;
        Q: length of attention query;
        KV: length of attention key and value.

        The shape of attention mask is (Q, KV).
        If self.use_memory is `True`:
          query = [right_context, utterance, summary];
          key, value = [memory, right_context, utterance];
          Q = R + U + S, KV = M + R + U.
        Otherwise:
          query = [right_context, utterance]
          key, value = [right_context, utterance]
          Q = R + U, KV = R + U.
        """
        U = utterance.size(0)
        num_chunks = math.ceil(U / self.chunk_length)

        right_context_mask = []
        utterance_mask = []
        summary_mask = []

        if self.use_memory:
            num_cols = 9
            # right context and utterance both attend to memory, right context,
            # utterance
            right_context_utterance_cols_mask = [
                idx in [1, 4, 7] for idx in range(num_cols)
            ]
            # summary attends to right context, utterance
            summary_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
            masks_to_concat = [right_context_mask, utterance_mask, summary_mask]
        else:
            num_cols = 6
            # right context and utterance both attend to right context and
            # utterance
            right_context_utterance_cols_mask = [
                idx in [1, 4] for idx in range(num_cols)
            ]
            summary_cols_mask = None
            masks_to_concat = [right_context_mask, utterance_mask]

        for chunk_idx in range(num_chunks):
            col_widths = self._gen_attention_mask_col_widths(chunk_idx, U)

            right_context_mask_block = _gen_attention_mask_block(
                col_widths,
                right_context_utterance_cols_mask,
                self.right_context_length,
                utterance.device,
            )
            right_context_mask.append(right_context_mask_block)

            utterance_mask_block = _gen_attention_mask_block(
                col_widths,
                right_context_utterance_cols_mask,
                min(
                    self.chunk_length,
                    U - chunk_idx * self.chunk_length,
                ),
                utterance.device,
            )
            utterance_mask.append(utterance_mask_block)

            if summary_cols_mask is not None:
                summary_mask_block = _gen_attention_mask_block(
                    col_widths, summary_cols_mask, 1, utterance.device
                )
                summary_mask.append(summary_mask_block)

        attention_mask = (
            1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])
        ).to(torch.bool)
        return attention_mask

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and non-streaming inference.

        B: batch size;
        D: input dimension;
        U: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (U + right_context_length, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, which contains the
            right_context at the end.

        Returns:
          A tuple of 2 tensors:
            - output utterance frames, with shape (U, B, D).
            - output_lengths, with shape (B,), without containing the
              right_context at the end.
        """
        U = x.size(0) - self.right_context_length
        x, pos_emb = self.encoder_pos(x, pos_len=U, neg_len=U)

        right_context = self._gen_right_context(x)
        utterance = x[:U]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        attention_mask = self._gen_attention_mask(utterance)
        memory = (
            self.init_memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[
                :-1
            ]
            if self.use_memory
            else torch.empty(0).to(dtype=x.dtype, device=x.device)
        )

        output = utterance
        for layer in self.emformer_layers:
            output, right_context, memory = layer(
                output,
                output_lengths,
                right_context,
                memory,
                attention_mask,
                pos_emb,
            )

        return output, output_lengths

    @torch.jit.export
    def infer(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
        conv_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]
    ]:
        """Forward pass for streaming inference.

        B: batch size;
        D: input dimension;
        U: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (U + right_context_length, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, which contains the
            right_context at the end.
          states (List[List[torch.Tensor]], optional):
            Cached states from proceeding chunk's computation, where each
            element (List[torch.Tensor]) corresponds to each emformer layer.
            (default: None)
          conv_caches (List[torch.Tensor], optional):
            Cached tensors of left context for causal convolution, where each
            element (Tensor) corresponds to each convolutional layer.
        Returns:
          (Tensor, Tensor, List[List[torch.Tensor]], List[torch.Tensor]):
            - output utterance frames, with shape (U, B, D).
            - output lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
            - updated convolution caches from current chunk.
        """
        assert x.size(0) == self.chunk_length + self.right_context_length, (
            "Per configured chunk_length and right_context_length, "
            f"expected size of {self.chunk_length + self.right_context_length} "
            f"for dimension 1 of x, but got {x.size(1)}."
        )

        pos_len = self.chunk_length + self.left_context_length
        neg_len = self.chunk_length
        x, pos_emb = self.encoder_pos(x, pos_len=pos_len, neg_len=neg_len)

        right_context_start_idx = x.size(0) - self.right_context_length
        right_context = x[right_context_start_idx:]
        utterance = x[:right_context_start_idx]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        memory = (
            self.init_memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            if self.use_memory
            else torch.empty(0).to(dtype=x.dtype, device=x.device)
        )
        output = utterance
        output_states: List[List[torch.Tensor]] = []
        output_conv_caches: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.emformer_layers):
            (
                output,
                right_context,
                memory,
                output_state,
                output_conv_cache,
            ) = layer.infer(
                output,
                output_lengths,
                right_context,
                memory,
                pos_emb,
                None if states is None else states[layer_idx],
                None if conv_caches is None else conv_caches[layer_idx],
            )
            output_states.append(output_state)
            output_conv_caches.append(output_conv_cache)

        return output, output_lengths, output_states, output_conv_caches


class Emformer(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        output_dim: int,
        chunk_length: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        cnn_module_kernel: int = 3,
        vgg_frontend: bool = False,
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        causal: bool = False,
    ):
        super().__init__()

        self.subsampling_factor = subsampling_factor
        self.right_context_length = right_context_length
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        if chunk_length % 4 != 0:
            raise NotImplementedError("chunk_length must be a mutiple of 4.")
        if left_context_length != 0 and left_context_length % 4 != 0:
            raise NotImplementedError(
                "left_context_length must be 0 or a mutiple of 4."
            )
        if right_context_length != 0 and right_context_length % 4 != 0:
            raise NotImplementedError(
                "right_context_length must be 0 or a mutiple of 4."
            )

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        if vgg_frontend:
            self.encoder_embed = VggSubsampling(num_features, d_model)
        else:
            self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder = EmformerEncoder(
            chunk_length // 4,
            d_model,
            nhead,
            dim_feedforward,
            num_encoder_layers,
            dropout,
            cnn_module_kernel,
            left_context_length=left_context_length // 4,
            right_context_length=right_context_length // 4,
            max_memory_size=max_memory_size,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
            causal=causal,
        )

        # TODO(fangjun): remove dropout
        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(d_model, output_dim)
        )

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and non-streaming inference.

        B: batch size;
        D: feature dimension;
        T: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (B, T, D).
          x_lens (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, containing the
            right_context at the end.

        Returns:
          (Tensor, Tensor):
            - output logits, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - logits lengths, with shape (B,), without containing the
              right_context at the end.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_lens = ((x_lens - 1) // 2 - 1) // 2
        assert x.size(0) == x_lens.max().item()

        output, output_lengths = self.encoder(x, x_lens)  # (T, N, C)

        logits = self.encoder_output_layer(output)
        logits = logits.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return logits, output_lengths

    @torch.jit.export
    def infer(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
        conv_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Forward pass for streaming inference.

        B: batch size;
        D: feature dimension;
        T: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (B, T, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, containing the
            right_context at the end.
          states (List[List[torch.Tensor]], optional):
            Cached states from proceeding chunk's computation, where each
            element (List[torch.Tensor]) corresponds to each emformer layer.
            (default: None)
          conv_caches (List[torch.Tensor], optional):
            Cached tensors of left context for causal convolution, where each
            element (Tensor) corresponds to each convolutional layer.
        Returns:
          (Tensor, Tensor):
            - output logits, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - logits lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
            - updated convolution caches from current chunk.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_lens = ((x_lens - 1) // 2 - 1) // 2
        assert x.size(0) == x_lens.max().item()

        (
            output,
            output_lengths,
            output_states,
            output_conv_caches,
        ) = self.encoder.infer(x, x_lens, states, conv_caches)

        logits = self.encoder_output_layer(output)
        logits = logits.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return logits, output_lengths, output_states, output_conv_caches


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py  # noqa

    Args:
      channels (int):
        The number of channels of conv layers.
      kernel_size (int):
        Kernerl size of conv layers.
      bias (bool):
        Whether to use bias in conv layers (default=True).
      causal (bool):
        Whether use causal convolution (default=False).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # from https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/convolution.py  # noqa
        if causal:
            self.left_padding = kernel_size - 1
            padding = 0
        else:
            self.left_padding = 0
            padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute convolution module.

        Args:
          x (torch.Tensor):
            Input tensor (#time, batch, channels).
          cache (torch.Tensor, optional):
            Cached tensor for left padding (#batch, channels, cache_time).
        Returns:
          A tuple of 2 tensors:
            - output tensor (#time, batch, channels).
            - updated cache tensor (#batch, channels, cache_time).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # 1D Depthwise Conv
        if self.left_padding > 0:
            # manualy padding self.lorder zeros to the left
            # make depthwise_conv causal
            if cache is None:
                x = nn.functional.pad(
                    x, (self.left_padding, 0), "constant", 0.0
                )
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                assert cache.size(2) == self.left_padding
                x = torch.cat([cache, x], dim=2)
            new_cache = x[:, :, x.size(2) - self.left_padding :]  # noqa
        else:
            # It's better we just return None if no cache is requried,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = None

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        x = self.depthwise_conv(x)
        # x is (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1), new_cache


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py  # noqa

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.pos_len = max_len
        self.neg_len = max_len
        self.gen_pe()

    def gen_pe(self) -> None:
        """Generate the positional encodings."""
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(self.pos_len, self.d_model)
        pe_negative = torch.zeros(self.neg_len, self.d_model)
        position_positive = torch.arange(
            0, self.pos_len, dtype=torch.float32
        ).unsqueeze(1)
        position_negative = torch.arange(
            0, self.neg_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position_positive * div_term)
        pe_positive[:, 1::2] = torch.cos(position_positive * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position_negative * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position_negative * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
        self.pe_positive = torch.flip(pe_positive, [0])
        self.pe_negative = pe_negative
        # self.pe = torch.cat([pe_positive, pe_negative], dim=1)

    def get_pe(
        self,
        pos_len: int,
        neg_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get positional encoding given positive length and negative length."""
        if self.pe_positive.dtype != dtype or str(
            self.pe_positive.device
        ) != str(device):
            self.pe_positive = self.pe_positive.to(dtype=dtype, device=device)
        if self.pe_negative.dtype != dtype or str(
            self.pe_negative.device
        ) != str(device):
            self.pe_negative = self.pe_negative.to(dtype=dtype, device=device)
        pe = torch.cat(
            [
                self.pe_positive[self.pos_len - pos_len :],
                self.pe_negative[1:neg_len],
            ],
            dim=0,
        )
        return pe

    def forward(
        self,
        x: torch.Tensor,
        pos_len: int,
        neg_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        x = x * self.xscale
        if pos_len > self.pos_len or neg_len > self.neg_len:
            self.pos_len = pos_len
            self.neg_len = neg_len
            self.gen_pe()
        pos_emb = self.get_pe(pos_len, neg_len, x.device, x.dtype)
        return self.dropout(x), self.dropout(pos_emb)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)
