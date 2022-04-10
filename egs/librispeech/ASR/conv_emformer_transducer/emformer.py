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
# It is modified based on
# https://github.com/pytorch/audio/blob/main/torchaudio/models/emformer.py.

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
      dropout (float, optional):
        Dropout probability. (Default: 0.0)
      tanh_on_mem (bool, optional):
        If ``True``, applies tanh to memory elements. (Default: ``False``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dropout: float = 0.0,
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
        self.dropout = dropout
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf

        self.scaling = (self.embed_dim // self.nhead) ** -0.5

        self.emb_to_key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.emb_to_query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb_to_key_value.weight)
        nn.init.constant_(self.emb_to_key_value.bias, 0.0)

        nn.init.xavier_uniform_(self.emb_to_query.weight)
        nn.init.constant_(self.emb_to_query.bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

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
        attention_probs = nn.functional.dropout(
            attention_probs, p=float(self.dropout), training=self.training
        )
        return attention_probs

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
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
        B = utterance.size(1)

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
            M = memory.size(0)
            R = right_context.size(0)
            right_context_end_idx = M + R
            key = torch.cat(
                [
                    key[:right_context_end_idx],
                    left_context_key,
                    key[right_context_end_idx:],
                ]
            )
            value = torch.cat(
                [
                    value[:right_context_end_idx],
                    left_context_val,
                    value[right_context_end_idx:],
                ]
            )

        # Compute attention weights from query, key, and value.
        reshaped_query, reshaped_key, reshaped_value = [
            tensor.contiguous()
            .view(-1, B * self.nhead, self.embed_dim // self.nhead)
            .transpose(0, 1)
            for tensor in [query, key, value]
        ]
        attention_weights = torch.bmm(
            reshaped_query * self.scaling, reshaped_key.transpose(1, 2)
        )

        # Compute padding mask
        if B == 1:
            padding_mask = None
        else:
            KV = key.size(0)
            U = utterance.size(0)
            padding_mask = make_pad_mask(KV - U + lengths)

        # Compute attention probabilities.
        attention_probs = self._gen_attention_probs(
            attention_weights, attention_mask, padding_mask
        )

        # Compute attention.
        attention = torch.bmm(attention_probs, reshaped_value)
        Q = query.size(0)
        assert attention.shape == (
            B * self.nhead,
            Q,
            self.embed_dim // self.nhead,
        )
        attention = (
            attention.transpose(0, 1).contiguous().view(Q, B, self.embed_dim)
        )

        # Apply output projection.
        outputs = self.out_proj(attention)

        S = summary.size(0)
        summary_start_idx = Q - S
        output_right_context_utterance = outputs[:summary_start_idx]
        output_memory = outputs[summary_start_idx:]
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
            utterance, lengths, right_context, summary, memory, attention_mask
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
            + right_context.size(0)
            + left_context_key.size(0)
            + utterance.size(0)
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
    ):
        super().__init__()

        self.attention = EmformerAttention(
            embed_dim=d_model,
            nhead=nhead,
            dropout=0.0,
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

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)

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

    def _apply_conv_module(
        self,
        right_context_utterance: torch.Tensor,
        right_context_end_idx: int,
    ) -> torch.Tensor:
        """Apply convolution module on utterance."""
        utterance = right_context_utterance[right_context_end_idx:]
        right_context = right_context_utterance[:right_context_end_idx]

        residual = utterance
        utterance = self.norm_conv(utterance)
        utterance = residual + self.dropout(self.conv_module(utterance))
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance

    def _apply_feed_forward_module(
        self,
        right_context_utterance: torch.Tensor,
    ) -> torch.Tensor:
        """Apply feed forward module."""
        residual = right_context_utterance
        right_context_utterance = self.norm_ff(right_context_utterance)
        right_context_utterance = residual + self.ff_scale * self.dropout(
            self.feed_forward(right_context_utterance)
        )
        return right_context_utterance

    def _apply_attention_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        right_context_end_idx: int,
        lengths: torch.Tensor,
        memory: torch.Tensor,
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
        )

        right_context_utterance = self._apply_conv_module(
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
        state: Optional[List[torch.Tensor]] = None,
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

         Returns:
           (Tensor, Tensor, List[torch.Tensor], Tensor):
             - output utterance, with shape (U, B, D);
             - output right_context, with shape (R, B, D);
             - output memory, with shape (1, B, D) or (0, B, D).
             - output state.
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
            state,
        )

        right_context_utterance = self._apply_conv_module(
            right_context_utterance, right_context_end_idx
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
        right_context = self._gen_right_context(x)
        utterance = x[: x.size(0) - self.right_context_length]
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
                output, output_lengths, right_context, memory, attention_mask
            )

        return output, output_lengths

    @torch.jit.export
    def infer(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
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
            element (List[torch.Tensor]) corresponding to each emformer layer.
            (default: None)

        Returns:
          (Tensor, Tensor, List[List[torch.Tensor]]):
            - output utterance frames, with shape (U, B, D).
            - output lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
        """
        assert x.size(0) == self.chunk_length + self.right_context_length, (
            "Per configured chunk_length and right_context_length, "
            f"expected size of {self.chunk_length + self.right_context_length} "
            f"for dimension 1 of x, but got {x.size(1)}."
        )
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
        for layer_idx, layer in enumerate(self.emformer_layers):
            output, right_context, memory, output_state = layer.infer(
                output,
                output_lengths,
                right_context,
                memory,
                None if states is None else states[layer_idx],
            )
            output_states.append(output_state)

        return output, output_lengths, output_states


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
            element (List[torch.Tensor]) corresponding to each emformer layer.
            (default: None)
        Returns:
          (Tensor, Tensor):
            - output logits, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - logits lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_lens = ((x_lens - 1) // 2 - 1) // 2
        assert x.size(0) == x_lens.max().item()

        output, output_lengths, output_states = self.encoder.infer(
            x, x_lens, states
        )  # (T, N, C)

        logits = self.encoder_output_layer(output)
        logits = logits.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return logits, output_lengths, output_states


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py  # noqa

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self, channels: int, kernel_size: int, bias: bool = True
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
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        # x is (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)
