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
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)

from icefall.utils import make_pad_mask


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py  # noqa

    Suppose:
      i -> position of query,
      j -> position of key(value),
    we use positive relative position embedding when key(value) is to the
    left of query(i.e., i > j) and negative embedding otherwise.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(
        self, d_model: int, dropout: float, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.pos_len = max_len
        self.neg_len = max_len
        self.gen_pe_positive()
        self.gen_pe_negative()

    def gen_pe_positive(self) -> None:
        """Generate the positive positional encodings."""
        pe_positive = torch.zeros(self.pos_len, self.d_model)
        position_positive = torch.arange(
            0, self.pos_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position_positive * div_term)
        pe_positive[:, 1::2] = torch.cos(position_positive * div_term)
        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
        self.pe_positive = torch.flip(pe_positive, [0])

    def gen_pe_negative(self) -> None:
        """Generate the negative positional encodings."""
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_negative = torch.zeros(self.neg_len, self.d_model)
        position_negative = torch.arange(
            0, self.neg_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_negative[:, 0::2] = torch.sin(-1 * position_negative * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position_negative * div_term)
        self.pe_negative = pe_negative

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
        """Scale input x and get positional encoding.
        Args:
            x (torch.Tensor): Input tensor (`*`).

        Returns:
          torch.Tensor:
            Encoded tensor of shape (`*`).
          torch.Tensor:
            Position embedding of shape (pos_len + neg_len - 1, `*`).
        """
        x = x * self.xscale
        if pos_len > self.pos_len:
            self.pos_len = pos_len
            self.gen_pe_positive()
        if neg_len > self.neg_len:
            self.neg_len = neg_len
            self.gen_pe_negative()
        pos_emb = self.get_pe(pos_len, neg_len, x.device, x.dtype)
        return self.dropout(x), self.dropout(pos_emb)


class ConvolutionModule(nn.Module):
    """ConvolutionModule.

    Modified from https://github.com/pytorch/audio/blob/main/torchaudio/prototype/models/conv_emformer.py # noqa

    Args:
      chunk_length (int):
        Length of each chunk.
      right_context_length (int):
        Length of right context.
      channels (int):
        The number of channels of conv layers.
      kernel_size (int):
        Kernerl size of conv layers.
      bias (bool):
        Whether to use bias in conv layers (default=True).
    """

    def __init__(
        self,
        chunk_length: int,
        right_context_length: int,
        channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.chunk_length = chunk_length
        self.right_context_length = right_context_length
        self.channels = channels

        self.pointwise_conv1 = ScaledConv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # After pointwise_conv1 we put x through a gated linear unit
        # (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in
        # the range 1 to 4, but sometimes, for some reason, for layer 0 the rms
        # ends up being very large, between 50 and 100 for different channels.
        # This will cause very peaky and sparse derivatives for the sigmoid
        # gating function, which will tend to make the loss function not learn
        # effectively.  (for most layers the average absolute values are in the
        # range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for
        # different layers, which likely breaks down as 0.5 for the "linear"
        # half and 0.2 to 0.3 for the part that goes into the sigmoid.
        # The idea is that if we constrain the rms values to a reasonable range
        # via a constraint of max_abs=10.0, it will be in a better position to
        # start learning something, i.e. to latch onto the correct range.
        self.deriv_balancer1 = ActivationBalancer(
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )

        # make it causal by padding cached (kernel_size - 1) frames on the left
        self.cache_size = kernel_size - 1
        self.depthwise_conv = ScaledConv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.25,
        )

    def _split_right_context(
        self,
        pad_utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          pad_utterance:
            Its shape is (cache_size + U, B, D).
          right_context:
            Its shape is (R, B, D).

        Returns:
          Right context segments padding with corresponding context.
          Its shape is (num_segs * B, D, cache_size + right_context_length).
        """
        U_, B, D = pad_utterance.size()
        R = right_context.size(0)
        assert self.right_context_length != 0
        assert R % self.right_context_length == 0
        num_chunks = R // self.right_context_length
        right_context = right_context.reshape(
            num_chunks, self.right_context_length, B, D
        )
        right_context = right_context.permute(0, 2, 1, 3).reshape(
            num_chunks * B, self.right_context_length, D
        )
        padding = []
        for idx in range(num_chunks):
            end_idx = min(U_, self.cache_size + (idx + 1) * self.chunk_length)
            start_idx = end_idx - self.cache_size
            padding.append(pad_utterance[start_idx:end_idx])
        padding = torch.cat(padding, dim=1).permute(1, 0, 2)
        # (num_segs * B, cache_size, D)
        pad_right_context = torch.cat([padding, right_context], dim=1)
        # (num_segs * B, cache_size + right_context_length, D)
        return pad_right_context.permute(0, 2, 1)

    def _merge_right_context(
        self, right_context: torch.Tensor, B: int
    ) -> torch.Tensor:
        """
        Args:
          right_context:
            Right context segments.
            It shape is (num_segs * B, D, right_context_length).
          B:
            Batch size.

        Returns:
          A tensor of shape (B, D, R), where
          R = num_segs * right_context_length.
        """
        right_context = right_context.reshape(
            -1, B, self.channels, self.right_context_length
        )
        right_context = right_context.permute(1, 2, 0, 3)
        right_context = right_context.reshape(B, self.channels, -1)
        return right_context

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal convolution module.

        Args:
          utterance (torch.Tensor):
            Utterance tensor of shape (U, B, D).
          right_context (torch.Tensor):
            Right context tensor of shape (R, B, D).
          cache (torch.Tensor, optional):
            Cached tensor for left padding of shape (B, D, cache_size).

        Returns:
          A tuple of 3 tensors:
            - output utterance of shape (U, B, D).
            - output right_context of shape (R, B, D).
            - updated cache tensor of shape (B, D, cache_size).
        """
        U, B, D = utterance.size()
        R, _, _ = right_context.size()

        # point-wise conv and GLU mechanism
        x = torch.cat([right_context, utterance], dim=0)  # (R + U, B, D)
        x = x.permute(1, 2, 0)  # (B, D, R + U)
        x = self.pointwise_conv1(x)  # (B, 2 * D, R + U)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (B, D, R + U)
        utterance = x[:, :, R:]  # (B, D, U)
        right_context = x[:, :, :R]  # (B, D, R)

        if cache is None:
            cache = torch.zeros(
                B, D, self.cache_size, device=x.device, dtype=x.dtype
            )
        else:
            assert cache.shape == (B, D, self.cache_size), cache.shape
        pad_utterance = torch.cat(
            [cache, utterance], dim=2
        )  # (B, D, cache + U)
        # update cache
        new_cache = pad_utterance[:, :, -self.cache_size :]

        # depth-wise conv on utterance
        utterance = self.depthwise_conv(pad_utterance)  # (B, D, U)

        if self.right_context_length > 0:
            # depth-wise conv on right_context
            pad_right_context = self._split_right_context(
                pad_utterance.permute(2, 0, 1), right_context.permute(2, 0, 1)
            )  # (num_segs * B, D, cache_size + right_context_length)
            right_context = self.depthwise_conv(
                pad_right_context
            )  # (num_segs * B, D, right_context_length)
            right_context = self._merge_right_context(
                right_context, B
            )  # (B, D, R)

        x = torch.cat([right_context, utterance], dim=2)  # (B, D, R + U)
        x = self.deriv_balancer2(x)
        x = self.activation(x)

        # point-wise conv
        x = self.pointwise_conv2(x)  # (B, D, R + U)

        right_context = x[:, :, :R]  # (B, D, R)
        utterance = x[:, :, R:]  # (B, D, U)
        return (
            utterance.permute(2, 0, 1),
            right_context.permute(2, 0, 1),
            new_cache,
        )

    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal convolution module applied on both utterance and right_context.

        Args:
          utterance (torch.Tensor):
            Utterance tensor of shape (U, B, D).
          right_context (torch.Tensor):
            Right context tensor of shape (R, B, D).
          cache (torch.Tensor, optional):
            Cached tensor for left padding of shape (B, D, cache_size).

        Returns:
          A tuple of 3 tensors:
            - output utterance of shape (U, B, D).
            - output right_context of shape (R, B, D).
            - updated cache tensor of shape (B, D, cache_size).
        """
        U, B, D = utterance.size()
        R, _, _ = right_context.size()

        # point-wise conv
        x = torch.cat([utterance, right_context], dim=0)  # (U + R, B, D)
        x = x.permute(1, 2, 0)  # (B, D, U + R)
        x = self.pointwise_conv1(x)  # (B, 2 * D, U + R)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (B, D, U + R)

        if cache is None:
            cache = torch.zeros(
                B, D, self.cache_size, device=x.device, dtype=x.dtype
            )
        else:
            assert cache.shape == (B, D, self.cache_size), cache.shape
        x = torch.cat([cache, x], dim=2)  # (B, D, cache_size + U + R)
        # update cache
        x_length = x.size(2)
        new_cache = x[:, :, x_length - R - self.cache_size : x_length - R]

        # 1-D depth-wise conv
        x = self.depthwise_conv(x)  # (B, D, U + R)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        # point-wise conv
        x = self.pointwise_conv2(x)  # (B, D, U + R)

        utterance = x[:, :, :U]  # (B, D, U)
        right_context = x[:, :, U:]  # (B, D, R)
        return (
            utterance.permute(2, 0, 1),
            right_context.permute(2, 0, 1),
            new_cache,
        )


class EmformerAttention(nn.Module):
    r"""Emformer layer attention module.

    Relative positional encoding is applied in this module, which is difference
    from https://github.com/pytorch/audio/blob/main/torchaudio/models/emformer.py  # noqa

    Args:
      embed_dim (int):
        Embedding dimension.
      nhead (int):
        Number of attention heads in each Emformer layer.
      chunk_length (int):
        Length of each input chunk.
      right_context_length (int):
        Length of right context.
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
        chunk_length: int,
        right_context_length: int,
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
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf
        self.head_dim = embed_dim // nhead
        self.chunk_length = chunk_length
        self.right_context_length = right_context_length
        self.dropout = dropout

        self.emb_to_key_value = ScaledLinear(
            embed_dim, 2 * embed_dim, bias=True
        )
        self.emb_to_query = ScaledLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = ScaledLinear(
            embed_dim, embed_dim, bias=True, initial_scale=0.25
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3  # noqa
        self.pos_bias_u = nn.Parameter(torch.Tensor(nhead, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(nhead, self.head_dim))
        self.pos_bias_u_scale = nn.Parameter(torch.zeros(()).detach())
        self.pos_bias_v_scale = nn.Parameter(torch.zeros(()).detach())

        self._reset_parameters()

    def _pos_bias_u(self):
        return self.pos_bias_u * self.pos_bias_u_scale.exp()

    def _pos_bias_v(self):
        return self.pos_bias_v * self.pos_bias_v_scale.exp()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_bias_u, std=0.01)
        nn.init.normal_(self.pos_bias_v, std=0.01)

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
            attention_probs, p=self.dropout, training=self.training
        )
        return attention_probs

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
          x: Input tensor, of shape (B, nhead, U, PE).
             U is the length of query vector.
             For training and validation mode,
               PE = 2 * U + right_context_length - 1.
             For inference mode,
               PE = tot_left_length + 2 * U + right_context_length - 1,
               where tot_left_length = M * chunk_length.

        Returns:
          A tensor of shape (B, nhead, U, out_len).
          For training and validation mode, out_len = U + right_context_length.
          For inference mode, out_len = tot_left_length + U + right_context_length.  # noqa
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

    def _get_right_context_part(
        self, matrix_bd_utterance: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          matrix_bd_utterance:
            (B * nhead, U, U + right_context_length)

        Returns:
          A tensor of shape (B * nhead, U, R),
          where R = num_chunks * right_context_length.
        """
        assert self.right_context_length > 0
        U = matrix_bd_utterance.size(1)
        num_chunks = math.ceil(U / self.chunk_length)
        right_context_blocks = []
        for i in range(num_chunks - 1):
            start_idx = (i + 1) * self.chunk_length
            end_idx = start_idx + self.right_context_length
            right_context_blocks.append(
                matrix_bd_utterance[:, :, start_idx:end_idx]
            )
        right_context_blocks.append(
            matrix_bd_utterance[:, :, -self.right_context_length :]
        )
        return torch.cat(right_context_blocks, dim=2)

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
        need_weights=False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Underlying chunk-wise attention implementation."""
        U, B, _ = utterance.size()
        R = right_context.size(0)
        M = memory.size(0)
        scaling = float(self.head_dim) ** -0.5

        # compute query with [right_context, utterance, summary].
        query = self.emb_to_query(
            torch.cat([right_context, utterance, summary])
        )
        # compute key and value with [memory, right_context, utterance].
        key, value = self.emb_to_key_value(
            torch.cat([memory, right_context, utterance])
        ).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            # now compute key and value with
            #   [memory, right context, left context, uttrance]
            # this is used in inference mode
            key = torch.cat([key[: M + R], left_context_key, key[M + R :]])
            value = torch.cat(
                [value[: M + R], left_context_val, value[M + R :]]
            )
        Q = query.size(0)
        KV = key.size(0)

        reshaped_key, reshaped_value = [
            tensor.contiguous()
            .view(KV, B * self.nhead, self.head_dim)
            .transpose(0, 1)
            for tensor in [key, value]
        ]  # both of shape (B * nhead, KV, head_dim)
        reshaped_query = (
            query.contiguous().view(Q, B, self.nhead, self.head_dim) * scaling
        )

        # compute attention score
        # first, compute attention matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3  # noqa
        query_with_bais_u = (
            (reshaped_query + self._pos_bias_u())
            .view(Q, B * self.nhead, self.head_dim)
            .transpose(0, 1)
        )  # (B * nhead, Q, head_dim)
        matrix_ac = torch.bmm(
            query_with_bais_u, reshaped_key.transpose(1, 2)
        )  # (B * nhead, Q, KV)

        # second, compute attention matrix b and matrix d
        # relative positional encoding is applied on the part of attention
        # between chunk (in query) and itself as well as its left context
        # (in key)
        utterance_with_bais_v = (
            reshaped_query[R : R + U] + self._pos_bias_v()
        ).permute(1, 2, 0, 3)
        # (B, nhead, U, head_dim)
        PE = pos_emb.size(0)
        if left_context_key is not None and left_context_val is not None:
            # inference mode
            L = left_context_key.size(0)
            tot_left_length = M * self.chunk_length if M > 0 else L
            assert tot_left_length >= L
            assert PE == tot_left_length + 2 * U + self.right_context_length - 1
        else:
            # training and validation mode
            assert PE == 2 * U + self.right_context_length - 1
        pos_emb = (
            self.linear_pos(pos_emb)
            .view(PE, self.nhead, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )  # (1, nhead, PE, head_dim)
        matrix_bd_utterance = torch.matmul(
            utterance_with_bais_v, pos_emb.transpose(-2, -1)
        )  # (B, nhead, U, PE)
        # rel-shift operation
        matrix_bd_utterance = self._rel_shift(matrix_bd_utterance)
        # (B, nhead, U, U + right_context_length) for training and validation mode; # noqa
        # (B, nhead, U, tot_left_length + U + right_context_length) for inference mode. # noqa
        matrix_bd_utterance = matrix_bd_utterance.contiguous().view(
            B * self.nhead, U, -1
        )
        matrix_bd = torch.zeros_like(matrix_ac)
        if left_context_key is not None and left_context_val is not None:
            # inference mode
            # key: [memory, right context, left context, utterance]
            # for memory
            if M > 0:
                # take average over the chunk frames for the memory vector
                matrix_bd[:, R : R + U, :M] = torch.nn.functional.avg_pool2d(
                    matrix_bd_utterance[:, :, :tot_left_length].unsqueeze(1),
                    kernel_size=(1, self.chunk_length),
                    stride=(1, self.chunk_length),
                ).squeeze(1)
            # for right_context
            if R > 0:
                matrix_bd[:, R : R + U, M : M + R] = matrix_bd_utterance[
                    :, :, tot_left_length + U :
                ]
            # for left_context and utterance
            matrix_bd[:, R : R + U, M + R :] = matrix_bd_utterance[
                :, :, tot_left_length - L : tot_left_length + U
            ]
        else:
            # training and validation mode
            # key: [memory, right context, utterance]
            # for memory
            if M > 0:
                # take average over the chunk frames for the memory vector
                matrix_bd[:, R : R + U, :M] = torch.nn.functional.avg_pool2d(
                    matrix_bd_utterance[:, :, :U].unsqueeze(1),
                    kernel_size=(1, self.chunk_length),
                    stride=(1, self.chunk_length),
                    ceil_mode=True,
                ).squeeze(1)[:, :, :-1]
            # for right_context
            if R > 0:
                matrix_bd[
                    :, R : R + U, M : M + R
                ] = self._get_right_context_part(matrix_bd_utterance)
            # for utterance
            matrix_bd[:, R : R + U, M + R :] = matrix_bd_utterance[:, :, :U]

        attention_weights = matrix_ac + matrix_bd

        # compute padding mask
        if B == 1:
            padding_mask = None
        else:
            padding_mask = make_pad_mask(KV - U + lengths)

        # compute attention probabilities
        attention_probs = self._gen_attention_probs(
            attention_weights, attention_mask, padding_mask
        )

        # compute attention outputs
        attention = torch.bmm(attention_probs, reshaped_value)
        assert attention.shape == (B * self.nhead, Q, self.head_dim)
        attention = (
            attention.transpose(0, 1).contiguous().view(Q, B, self.embed_dim)
        )

        # apply output projection
        outputs = self.out_proj(attention)

        output_right_context_utterance = outputs[: R + U]
        output_memory = outputs[R + U :]
        if self.tanh_on_mem:
            output_memory = torch.tanh(output_memory)
        else:
            output_memory = torch.clamp(output_memory, min=-10, max=10)

        if need_weights:
            # average over attention heads
            attention_probs = attention_probs.reshape(B, self.nhead, Q, KV)
            attention_probs = attention_probs.sum(dim=1) / self.nhead
            probs_memory = attention_probs[:, R : R + U, :M].sum(dim=2)
            probs_frames = attention_probs[:, R : R + U, M:].sum(dim=2)
            return (
                output_right_context_utterance,
                output_memory,
                key,
                value,
                probs_memory,
                probs_frames,
            )

        return (
            output_right_context_utterance,
            output_memory,
            key,
            value,
            None,
            None,
        )

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
        """Forward pass for training and validation mode.

        B: batch size;
        D: embedding dimension;
        R: length of the hard-copied right contexts;
        U: length of full utterance;
        S: length of summary vectors;
        M: length of memory vectors.

        It computes a `big` attention matrix on full utterance and
        then utilizes a pre-computed mask to simulate chunk-wise attention.

        It concatenates three blocks: hard-copied right contexts,
        full utterance, and summary vectors, as a `big` block,
        to compute the query tensor:
        query = [right_context, utterance, summary],
        with length Q = R + U + S.
        It concatenates the three blocks: memory vectors,
        hard-copied right contexts, and full utterance as another `big` block,
        to compute the key and value tensors:
        key & value = [memory, right_context, utterance],
        with length KV = M + R + U.
        Attention scores is computed with above `big` query and key.

        Then the underlying chunk-wise attention is obtained by applying
        the attention mask. Suppose
        c_i: chunk at index i;
        r_i: right context that c_i can use;
        l_i: left context that c_i can use;
        m_i: past memory vectors from previous layer that c_i can use;
        s_i: summary vector of c_i;
        The target chunk-wise attention is:
        c_i, r_i (in query) -> l_i, c_i, r_i, m_i (in key);
        s_i (in query) -> l_i, c_i, r_i (in key).

        Relative positional encoding is applied on the part of attention between
        [utterance] (in query) and [memory, right_context, utterance] (in key).
        Actually, it is applied on the part of attention between each chunk
        (in query) and itself, its memory vectors, left context, and right
        context (in key), after applying the mask:
        c_i (in query) -> l_i, c_i, r_i, m_i (in key).

        Args:
          utterance (torch.Tensor):
            Full utterance frames, with shape (U, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing
            number of valid frames for i-th batch element in utterance.
          right_context (torch.Tensor):
            Hard-copied right context frames, with shape (R, B, D),
            where R = num_chunks * right_context_length
          summary (torch.Tensor):
            Summary elements with shape (S, B, D), where S = num_chunks.
            It is an empty tensor without using memory.
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D), where M = num_chunks - 1.
            It is an empty tensor without using memory.
          attention_mask (torch.Tensor):
            Pre-computed attention mask to simulate underlying chunk-wise
            attention, with shape (Q, KV).
          pos_emb (torch.Tensor):
            Position encoding embedding, with shape (PE, D).
            where PE = 2 * U + right_context_length - 1.

        Returns:
          A tuple containing 2 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (M, B, D), where M = S - 1 or M = 0.
            - summary of attention weights on memory, with shape (B, U).
            - summary of attention weights on left context, utterance, and
              right context, with shape (B, U).
        """
        (
            output_right_context_utterance,
            output_memory,
            _,
            _,
            probs_memory,
            probs_frames,
        ) = self._forward_impl(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            attention_mask,
            pos_emb,
            need_weights=True,
        )
        return (
            output_right_context_utterance,
            output_memory[:-1],
            probs_memory,
            probs_frames,
        )

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
        R: length of right context;
        U: length of utterance, i.e., current chunk;
        L: length of cached left context;
        S: length of summary vectors, S = 1;
        M: length of cached memory vectors.

        It concatenates the right context, utterance (i.e., current chunk)
        and summary vector of current chunk, to compute the query tensor:
        query = [right_context, utterance, summary],
        with length Q = R + U + S.
        It concatenates the memory vectors, right context, left context, and
        current chunk, to compute the key and value tensors:
        key & value = [memory, right_context, left_context, utterance],
        with length KV = M + R + L + U.

        The chunk-wise attention is:
        chunk, right context (in query) ->
          left context, chunk, right context, memory vectors (in key);
        summary (in query) -> left context, chunk, right context (in key).

        Relative positional encoding is applied on the part of attention:
        chunk (in query) ->
          left context, chunk, right context, memory vectors (in key);

        Args:
          utterance (torch.Tensor):
            Current chunk frames, with shape (U, B, D), where U = chunk_length.
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing
            number of valid frames for i-th batch element in utterance.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D),
            where R = right_context_length.
          summary (torch.Tensor):
            Summary vector with shape (1, B, D), or empty tensor.
          memory (torch.Tensor):
            Memory vectors, with shape (M, B, D), or empty tensor.
          left_context_key (torch,Tensor):
            Cached attention key of left context from preceding computation,
            with shape (L, B, D), where L <= left_context_length.
          left_context_val (torch.Tensor):
            Cached attention value of left context from preceding computation,
            with shape (L, B, D), where L <= left_context_length.
          pos_emb (torch.Tensor):
            Position encoding embedding, with shape (PE, D),
            where PE = M * chunk_length + 2 * U - 1 if M > 0 else L + 2 * U - 1.

        Returns:
          A tuple containing 4 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (1, B, D) or (0, B, D).
            - attention key of left context and utterance, which would be cached
              for next computation, with shape (L + U, B, D).
            - attention value of left context and utterance, which would be
              cached for next computation, with shape (L + U, B, D).
        """
        U = utterance.size(0)
        R = right_context.size(0)
        L = left_context_key.size(0)
        S = summary.size(0)
        M = memory.size(0)

        # query = [right context, utterance, summary]
        Q = R + U + S
        # key, value = [memory, right context, left context, uttrance]
        KV = M + R + L + U
        attention_mask = torch.zeros(Q, KV).to(
            dtype=torch.bool, device=utterance.device
        )
        # disallow attention bettween the summary vector with the memory bank
        attention_mask[-1, :M] = True
        (
            output_right_context_utterance,
            output_memory,
            key,
            value,
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
            left_context_key=left_context_key,
            left_context_val=left_context_val,
        )
        return (
            output_right_context_utterance,
            output_memory,
            key[M + R :],
            value[M + R :],
        )


class EmformerEncoderLayer(nn.Module):
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
      layer_dropout (float, optional):
        Layer dropout probability. (Default: 0.0)
      cnn_module_kernel (int):
        Kernel size of convolution module.
      left_context_length (int, optional):
        Length of left context. (Default: 0)
      right_context_length (int, optional):
        Length of right context. (Default: 0)
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
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.attention = EmformerAttention(
            embed_dim=d_model,
            nhead=nhead,
            chunk_length=chunk_length,
            right_context_length=right_context_length,
            dropout=dropout,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )
        self.summary_op = nn.AvgPool1d(
            kernel_size=chunk_length, stride=chunk_length, ceil_mode=True
        )

        self.feed_forward_macaron = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.conv_module = ConvolutionModule(
            chunk_length,
            right_context_length,
            d_model,
            cnn_module_kernel,
        )

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean
        # (or at least, zero-median).
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout)

        self.layer_dropout = layer_dropout
        self.left_context_length = left_context_length
        self.chunk_length = chunk_length
        self.right_context_length = right_context_length
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

    def _apply_conv_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
    ) -> torch.Tensor:
        """Apply convolution module in training and validation mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]
        utterance, right_context, _ = self.conv_module(utterance, right_context)
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance

    def _apply_conv_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        conv_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply convolution module on utterance in inference mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]
        utterance, right_context, conv_cache = self.conv_module.infer(
            utterance, right_context, conv_cache
        )
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance, conv_cache

    def _apply_attention_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        lengths: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention module in training and validation mode."""
        if attention_mask is None:
            raise ValueError(
                "attention_mask must be not None in training or validation mode."  # noqa
            )
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]

        if self.use_memory:
            summary = self.summary_op(utterance.permute(1, 2, 0)).permute(
                2, 0, 1
            )
        else:
            summary = torch.empty(0).to(
                dtype=utterance.dtype, device=utterance.device
            )
        output_right_context_utterance, output_memory, _, _ = self.attention(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            memory=memory,
            attention_mask=attention_mask,
            pos_emb=pos_emb,
        )

        return output_right_context_utterance, output_memory

    def _apply_attention_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        lengths: torch.Tensor,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Apply attention module in inference mode.
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
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]

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
        U = utterance.size(0)
        # pos_emb is of shape [PE, D], where PE = M * chunk_length + 2 * U - 1,
        # for query of [utterance] (i), key-value [memory vectors, left context, utterance, right context] (j)  # noqa
        # the max relative distance i - j is M * chunk_length + U - 1
        # the min relative distance i - j is -(U + right_context_length - 1)
        M = pre_memory.size(0)  # M <= max_memory_size
        if self.max_memory_size > 0:
            PE = M * self.chunk_length + 2 * U + self.right_context_length - 1
            tot_PE = (
                self.max_memory_size * self.chunk_length
                + 2 * U
                + self.right_context_length
                - 1
            )
        else:
            L = left_context_key.size(0)
            PE = L + 2 * U + self.right_context_length - 1
            tot_PE = (
                self.left_context_length + 2 * U + self.right_context_length - 1
            )
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
        state = self._pack_state(
            next_key, next_val, utterance.size(0), memory, state
        )
        return output_right_context_utterance, output_memory, state

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training and validation mode.

        B: batch size;
        D: embedding dimension;
        R: length of hard-copied right contexts;
        U: length of full utterance;
        M: length of memory vectors.

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
            It is an empty tensor without using memory.
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
        R = right_context.size(0)
        src = torch.cat([right_context, utterance])
        src_orig = src

        warmup_scale = min(0.1 + warmup, 1.0)
        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # emformer attention module
        src_att, output_memory = self._apply_attention_module_forward(
            src, R, lengths, memory, pos_emb, attention_mask
        )
        src = src + self.dropout(src_att)

        # convolution module
        src_conv = self._apply_conv_module_forward(src, R)
        src = src + self.dropout(src_conv)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        output_utterance = src[R:]
        output_right_context = src[:R]
        return output_utterance, output_right_context, output_memory

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
        R = right_context.size(0)
        src = torch.cat([right_context, utterance])

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # emformer attention module
        (
            src_att,
            output_memory,
            output_state,
        ) = self._apply_attention_module_infer(
            src, R, lengths, memory, pos_emb, state
        )
        src = src + self.dropout(src_att)

        # convolution module
        src_conv, conv_cache = self._apply_conv_module_infer(src, R, conv_cache)
        src = src + self.dropout(src_conv)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        output_utterance = src[R:]
        output_right_context = src[:R]
        return (
            output_utterance,
            output_right_context,
            output_memory,
            output_state,
            conv_cache,
        )


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
      layer_dropout (float, optional):
        Layer dropout probability. (default: 0.0)
      cnn_module_kernel (int):
        Kernel size of convolution module.
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
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
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
                EmformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    chunk_length=chunk_length,
                    dropout=dropout,
                    layer_dropout=layer_dropout,
                    cnn_module_kernel=cnn_module_kernel,
                    left_context_length=left_context_length,
                    right_context_length=right_context_length,
                    max_memory_size=max_memory_size,
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        self.max_memory_size = max_memory_size

    def _gen_right_context(self, x: torch.Tensor) -> torch.Tensor:
        """Hard copy each chunk's right context and concat them."""
        T = x.shape[0]
        num_chunks = math.ceil(
            (T - self.right_context_length) / self.chunk_length
        )
        right_context_blocks = []
        for seg_idx in range(num_chunks - 1):
            start = (seg_idx + 1) * self.chunk_length
            end = start + self.right_context_length
            right_context_blocks.append(x[start:end])
        right_context_blocks.append(x[T - self.right_context_length :])
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
        """Generate attention mask to simulate underlying chunk-wise attention
        computation, where chunk-wise connections are filled with `False`,
        and other unnecessary connections beyond chunk are filled with `True`.

        R: length of hard-copied right contexts;
        U: length of full utterance;
        S: length of summary vectors;
        M: length of memory vectors;
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

        Suppose:
          c_i: chunk at index i;
          r_i: right context that c_i can use;
          l_i: left context that c_i can use;
          m_i: past memory vectors from previous layer that c_i can use;
          s_i: summary vector of c_i.
        The target chunk-wise attention is:
          c_i, r_i (in query) -> l_i, c_i, r_i, m_i (in key);
          s_i (in query) -> l_i, c_i, r_i (in key).
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
        self, x: torch.Tensor, lengths: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and validation mode.

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
        x, pos_emb = self.encoder_pos(
            x, pos_len=U, neg_len=U + self.right_context_length
        )

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
                warmup=warmup,
            )

        return output, output_lengths

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

        pos_len = (
            self.max_memory_size * self.chunk_length + self.chunk_length
            if self.max_memory_size > 0
            else self.left_context_length + self.chunk_length
        )
        neg_len = self.chunk_length + self.right_context_length
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
        chunk_length: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 3,
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
        if (
            max_memory_size > 0
            and max_memory_size * chunk_length < left_context_length
        ):
            raise NotImplementedError(
                "max_memory_size * chunk_length can not be less than left_context_length"  # noqa
            )

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder = EmformerEncoder(
            chunk_length=chunk_length // 4,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
            cnn_module_kernel=cnn_module_kernel,
            left_context_length=left_context_length // 4,
            right_context_length=right_context_length // 4,
            max_memory_size=max_memory_size,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
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
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          (Tensor, Tensor):
            - output embedding, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - output lengths, with shape (B,), without containing the
              right_context at the end.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_lens = ((x_lens - 1) // 2 - 1) // 2
        assert x.size(0) == x_lens.max().item()

        output, output_lengths = self.encoder(
            x, x_lens, warmup=warmup
        )  # (T, N, C)

        output = output.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return output, output_lengths

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
            - output embedding, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - output lengths, with shape (B,), without containing the
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

        output = output.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return output, output_lengths, output_states, output_conv_caches


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
        """
        assert in_channels >= 7
        super().__init__()

        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=1,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.out = ScaledLinear(
            layer3_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        self.out_norm = BasicNorm(out_channels, learn_eps=False)
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x
