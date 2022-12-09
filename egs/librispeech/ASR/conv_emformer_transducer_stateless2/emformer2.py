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
# 1) https://github.com/pytorch/audio/blob/main/torchaudio/models/emformer.py  # noqa
# 2) https://github.com/pytorch/audio/blob/main/torchaudio/prototype/models/conv_emformer.py  # noqa

import math
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

LOG_EPSILON = math.log(1e-10)


def unstack_states(
    states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
) -> List[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]]:
    """Unstack the emformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Args:
      states:
        A tuple of 2 elements.
        ``states[0]`` is the attention caches of a batch of utterance.
        ``states[1]`` is the convolution caches of a batch of utterance.
        ``len(states[0])`` and ``len(states[1])`` both eqaul to number of layers.  # noqa

    Returns:
      A list of states.
      ``states[i]`` is a tuple of 2 elements of i-th utterance.
      ``states[i][0]`` is the attention caches of i-th utterance.
      ``states[i][1]`` is the convolution caches of i-th utterance.
      ``len(states[i][0])`` and ``len(states[i][1])`` both eqaul to number of layers.  # noqa
    """

    attn_caches, conv_caches = states
    batch_size = conv_caches[0].size(0)
    num_layers = len(attn_caches)

    list_attn_caches = [None] * batch_size
    for i in range(batch_size):
        list_attn_caches[i] = [[] for _ in range(num_layers)]
    for li, layer in enumerate(attn_caches):
        for s in layer:
            s_list = s.unbind(dim=1)
            for bi, b in enumerate(list_attn_caches):
                b[li].append(s_list[bi])

    list_conv_caches = [None] * batch_size
    for i in range(batch_size):
        list_conv_caches[i] = [None] * num_layers
    for li, layer in enumerate(conv_caches):
        c_list = layer.unbind(dim=0)
        for bi, b in enumerate(list_conv_caches):
            b[li] = c_list[bi]

    ans = [None] * batch_size
    for i in range(batch_size):
        ans[i] = [list_attn_caches[i], list_conv_caches[i]]

    return ans


def stack_states(
    state_list: List[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]]
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
    """Stack list of emformer states that correspond to separate utterances
    into a single emformer state so that it can be used as an input for
    emformer when those utterances are formed into a batch.

    Note:
      It is the inverse of :func:`unstack_states`.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the emformer model for a single utterance.
        ``states[i]`` is a tuple of 2 elements of i-th utterance.
        ``states[i][0]`` is the attention caches of i-th utterance.
        ``states[i][1]`` is the convolution caches of i-th utterance.
        ``len(states[i][0])`` and ``len(states[i][1])`` both eqaul to number of layers.  # noqa

    Returns:
      A new state corresponding to a batch of utterances.
      See the input argument of :func:`unstack_states` for the meaning
      of the returned tensor.
    """
    batch_size = len(state_list)

    attn_caches = []
    for layer in state_list[0][0]:
        if batch_size > 1:
            # Note: We will stack attn_caches[layer][s][] later to get attn_caches[layer][s]  # noqa
            attn_caches.append([[s] for s in layer])
        else:
            attn_caches.append([s.unsqueeze(1) for s in layer])
    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states[0]):
            for si, s in enumerate(layer):
                attn_caches[li][si].append(s)
                if b == batch_size - 1:
                    attn_caches[li][si] = torch.stack(attn_caches[li][si], dim=1)

    conv_caches = []
    for layer in state_list[0][1]:
        if batch_size > 1:
            # Note: We will stack conv_caches[layer][] later to get conv_caches[layer]  # noqa
            conv_caches.append([layer])
        else:
            conv_caches.append(layer.unsqueeze(0))
    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states[1]):
            conv_caches[li].append(layer)
            if b == batch_size - 1:
                conv_caches[li] = torch.stack(conv_caches[li], dim=0)

    return [attn_caches, conv_caches]


class ConvolutionModule(nn.Module):
    """ConvolutionModule.

    Modified from https://github.com/pytorch/audio/blob/main/torchaudio/prototype/models/conv_emformer.py # noqa

    Args:
      chunk_length (int):
        Length of each chunk.
      right_context_length (int):
        Length of right context.
      channels (int):
        The number of input channels and output channels of conv layers.
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
        super().__init__()
        # kernerl_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0, kernel_size

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

        intervals = torch.arange(
            0, self.chunk_length * (num_chunks - 1), self.chunk_length
        )
        first = torch.arange(self.chunk_length, self.chunk_length + self.cache_size)
        indexes = intervals.unsqueeze(1) + first.unsqueeze(0)
        indexes = torch.cat(
            [indexes, torch.arange(U_ - self.cache_size, U_).unsqueeze(0)]
        )
        padding = pad_utterance[indexes]  # (num_chunks, cache_size, B, D)
        padding = padding.permute(0, 2, 1, 3).reshape(
            num_chunks * B, self.cache_size, D
        )

        pad_right_context = torch.cat([padding, right_context], dim=1)
        # (num_chunks * B, cache_size + right_context_length, D)
        return pad_right_context.permute(0, 2, 1)

    def _merge_right_context(self, right_context: torch.Tensor, B: int) -> torch.Tensor:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal convolution module.

        Args:
          utterance (torch.Tensor):
            Utterance tensor of shape (U, B, D).
          right_context (torch.Tensor):
            Right context tensor of shape (R, B, D).

        Returns:
          A tuple of 2 tensors:
          - output utterance of shape (U, B, D).
          - output right_context of shape (R, B, D).
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

        # make causal convolution
        cache = torch.zeros(B, D, self.cache_size, device=x.device, dtype=x.dtype)
        pad_utterance = torch.cat([cache, utterance], dim=2)  # (B, D, cache + U)

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
            right_context = self._merge_right_context(right_context, B)  # (B, D, R)

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
        )

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        #  U, B, D = utterance.size()
        #  R, _, _ = right_context.size()
        U = self.chunk_length
        B = 1
        D = self.channels
        R = self.right_context_length

        # point-wise conv
        x = torch.cat([utterance, right_context], dim=0)  # (U + R, B, D)
        x = x.permute(1, 2, 0)  # (B, D, U + R)
        x = self.pointwise_conv1(x)  # (B, 2 * D, U + R)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (B, D, U + R)

        # make causal convolution
        assert cache.shape == (B, D, self.cache_size), cache.shape
        x = torch.cat([cache, x], dim=2)  # (B, D, cache_size + U + R)
        # update cache
        new_cache = x[:, :, -R - self.cache_size : -R]

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
        left_context_length: int,
        chunk_length: int,
        right_context_length: int,
        memory_size: int,
        dropout: float = 0.0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if embed_dim % nhead != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) is not a multiple of nhead ({nhead})."
            )

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf
        self.head_dim = embed_dim // nhead
        self.dropout = dropout

        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        self.memory_size = memory_size

        self.emb_to_key_value = ScaledLinear(embed_dim, 2 * embed_dim, bias=True)
        self.emb_to_query = ScaledLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = ScaledLinear(
            embed_dim, embed_dim, bias=True, initial_scale=0.25
        )

    def _gen_attention_probs(
        self,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
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
            attention_weights_float = attention_weights_float.view(B, self.nhead, Q, -1)
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

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Underlying chunk-wise attention implementation."""
        #  U, B, _ = utterance.size()
        #  R = right_context.size(0)
        #  M = memory.size(0)

        U = self.chunk_length
        B = 1
        R = self.right_context_length
        M = self.memory_size
        L = self.left_context_length

        scaling = float(self.head_dim) ** -0.5

        # compute query with [right_context, utterance].
        query = self.emb_to_query(torch.cat([right_context, utterance]))
        # compute key and value with [memory, right_context, utterance].
        key, value = self.emb_to_key_value(
            torch.cat([memory, right_context, utterance])
        ).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            # now compute key and value with
            #   [memory, right context, left context, uttrance]
            # this is used in inference mode
            key = torch.cat([key[: M + R], left_context_key, key[M + R :]])
            value = torch.cat([value[: M + R], left_context_val, value[M + R :]])

        #  Q = query.size(0)
        Q = U + R

        # KV = key.size(0)

        reshaped_query = query.view(Q, self.nhead, self.head_dim).permute(1, 0, 2)
        reshaped_key = key.view(M + R + U + L, self.nhead, self.head_dim).permute(
            1, 0, 2
        )
        reshaped_value = value.view(M + R + U + L, self.nhead, self.head_dim).permute(
            1, 0, 2
        )

        #  reshaped_query, reshaped_key, reshaped_value = [
        #      tensor.contiguous().view(-1, B * self.nhead, self.head_dim).transpose(0, 1)
        #      for tensor in [query, key, value]
        #  ]  # (B * nhead, Q or KV, head_dim)
        attention_weights = torch.bmm(
            reshaped_query * scaling, reshaped_key.permute(0, 2, 1)
        )  # (B * nhead, Q, KV)

        # compute attention probabilities
        if False:
            attention_probs = self._gen_attention_probs(
                attention_weights, attention_mask, padding_mask
            )
        else:
            attention_probs = nn.functional.softmax(attention_weights, dim=-1)

        # compute attention outputs
        attention = torch.bmm(attention_probs, reshaped_value)
        assert attention.shape == (B * self.nhead, Q, self.head_dim)
        attention = attention.permute(1, 0, 2).reshape(-1, self.embed_dim)
        # TODO(fangjun): ncnn does not support reshape(-1, 1, self.embed_dim)
        # We have to change InnerProduct in ncnn to ignore the extra dim below
        attention = attention.unsqueeze(1)

        # apply output projection
        output_right_context_utterance = self.out_proj(attention)
        # The return shape of output_right_context_utterance is (10, 1, 512)

        return output_right_context_utterance, key, value

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Modify docs.
        """Forward pass for training and validation mode.

        B: batch size;
        D: embedding dimension;
        R: length of the hard-copied right contexts;
        U: length of full utterance;
        M: length of memory vectors.

        It computes a `big` attention matrix on full utterance and
        then utilizes a pre-computed mask to simulate chunk-wise attention.

        It concatenates three blocks: hard-copied right contexts,
        and full utterance, as a `big` block,
        to compute the query tensor:
        query = [right_context, utterance],
        with length Q = R + U.
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
        The target chunk-wise attention is:
        c_i, r_i (in query) -> l_i, c_i, r_i, m_i (in key)

        Args:
          utterance (torch.Tensor):
            Full utterance frames, with shape (U, B, D).
          right_context (torch.Tensor):
            Hard-copied right context frames, with shape (R, B, D),
            where R = num_chunks * right_context_length
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D), where M = num_chunks - 1.
            It is an empty tensor without using memory.
          attention_mask (torch.Tensor):
            Pre-computed attention mask to simulate underlying chunk-wise
            attention, with shape (Q, KV).
          padding_mask (torch.Tensor):
            Padding mask of key tensor, with shape (B, KV).

        Returns:
          Output of right context and utterance, with shape (R + U, B, D).
        """
        output_right_context_utterance, _, _ = self._forward_impl(
            utterance,
            right_context,
            memory,
            attention_mask,
            padding_mask=padding_mask,
        )
        return output_right_context_utterance

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        left_context_key: torch.Tensor,
        left_context_val: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        B: batch size;
        D: embedding dimension;
        R: length of right context;
        U: length of utterance, i.e., current chunk;
        L: length of cached left context;
        M: length of cached memory vectors.

        It concatenates the right context and utterance (i.e., current chunk)
        of current chunk, to compute the query tensor:
        query = [right_context, utterance],
        with length Q = R + U.
        It concatenates the memory vectors, right context, left context, and
        current chunk, to compute the key and value tensors:
        key & value = [memory, right_context, left_context, utterance],
        with length KV = M + R + L + U.

        The chunk-wise attention is:
        chunk, right context (in query) ->
          left context, chunk, right context, memory vectors (in key).

        Args:
          utterance (torch.Tensor):
            Current chunk frames, with shape (U, B, D), where U = chunk_length.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D),
            where R = right_context_length.
          memory (torch.Tensor):
            Memory vectors, with shape (M, B, D), or empty tensor.
          left_context_key (torch,Tensor):
            Cached attention key of left context from preceding computation,
            with shape (L, B, D).
          left_context_val (torch.Tensor):
            Cached attention value of left context from preceding computation,
            with shape (L, B, D).
          padding_mask (torch.Tensor):
            Padding mask of key tensor, with shape (B, KV).

        Returns:
          A tuple containing 4 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - attention key of left context and utterance, which would be cached
              for next computation, with shape (L + U, B, D).
            - attention value of left context and utterance, which would be
              cached for next computation, with shape (L + U, B, D).
        """
        #  U = utterance.size(0)
        #  R = right_context.size(0)
        #  L = left_context_key.size(0)
        #  M = memory.size(0)

        U = self.chunk_length
        R = self.right_context_length
        L = self.left_context_length
        M = self.memory_size

        # query = [right context, utterance]
        Q = R + U
        # key, value = [memory, right context, left context, utterance]
        KV = M + R + L + U
        attention_mask = torch.zeros(Q, KV).to(
            dtype=torch.bool, device=utterance.device
        )

        output_right_context_utterance, key, value = self._forward_impl(
            utterance,
            right_context,
            memory,
            attention_mask,
            padding_mask=padding_mask,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
        )
        return (
            output_right_context_utterance,
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
      memory_size (int, optional):
        Number of memory elements to use. (Default: 0)
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
        memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.attention = EmformerAttention(
            embed_dim=d_model,
            nhead=nhead,
            left_context_length=left_context_length,
            chunk_length=chunk_length,
            memory_size=memory_size,
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
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        self.memory_size = memory_size
        self.d_model = d_model
        self.use_memory = memory_size > 0

    def _update_attn_cache(
        self,
        next_key: torch.Tensor,
        next_val: torch.Tensor,
        memory: torch.Tensor,
        attn_cache: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Update cached attention state:
        1) output memory of current chunk in the lower layer;
        2) attention key and value in current chunk's computation, which would
        be reused in next chunk's computation.
        """
        # attn_cache[0].shape (self.memory_size, 1, 512)
        # memory.shape (1, 1, 512)
        # attn_cache[1].shape (self.left_context_length, 1, 512)
        # attn_cache[2].shape (self.left_context_length, 1, 512)
        # next_key.shape (self.left_context_length + self.right_context_utterance, 1, 512)
        # next_value.shape (self.left_context_length + self.right_context_utterance, 1, 512)
        new_memory = torch.cat([attn_cache[0], memory])
        # TODO(fangjun): Remove torch.cat
        #  new_key = torch.cat([attn_cache[1], next_key])
        #  new_val = torch.cat([attn_cache[2], next_val])
        attn_cache[0] = new_memory[1:]
        attn_cache[1] = next_key[-self.left_context_length :]
        attn_cache[2] = next_val[-self.left_context_length :]
        return attn_cache

    def _apply_conv_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
    ) -> torch.Tensor:
        """Apply convolution module in training and validation mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]
        utterance, right_context = self.conv_module(utterance, right_context)
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance

    def _apply_conv_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        conv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention module in training and validation mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]

        if self.use_memory:
            memory = self.summary_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[
                :-1, :, :
            ]
        else:
            memory = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        output_right_context_utterance = self.attention(
            utterance=utterance,
            right_context=right_context,
            memory=memory,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )

        return output_right_context_utterance

    def _apply_attention_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        attn_cache: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Apply attention module in inference mode.
        1) Unpack cached states including:
           - memory from previous chunks;
           - attention key and value of left context from preceding
             chunk's compuation;
        2) Apply attention computation;
        3) Update cached attention states including:
           - memory of current chunk;
           - attention key and value in current chunk's computation, which would
             be resued in next chunk's computation.
        """
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]

        pre_memory = attn_cache[0]
        left_context_key = attn_cache[1]
        left_context_val = attn_cache[2]

        if self.use_memory:
            memory = torch.mean(utterance, dim=0, keepdim=True)

            #  memory = self.summary_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[
            #          :1, :, :
            #  ]
        else:
            memory = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        (output_right_context_utterance, next_key, next_val) = self.attention.infer(
            utterance=utterance,
            right_context=right_context,
            memory=pre_memory,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
            padding_mask=padding_mask,
        )
        attn_cache = self._update_attn_cache(next_key, next_val, memory, attn_cache)
        return output_right_context_utterance, attn_cache

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training and validation mode.

        B: batch size;
        D: embedding dimension;
        R: length of hard-copied right contexts;
        U: length of full utterance;
        M: length of memory vectors.

        Args:
          utterance (torch.Tensor):
            Utterance frames, with shape (U, B, D).
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D).
          attention_mask (torch.Tensor):
            Attention mask for underlying attention module,
            with shape (Q, KV), where Q = R + U, KV = M + R + U.
          padding_mask (torch.Tensor):
            Padding mask of ker tensor, with shape (B, KV).

        Returns:
          A tuple containing 2 tensors:
            - output utterance, with shape (U, B, D).
            - output right context, with shape (R, B, D).
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
        src_att = self._apply_attention_module_forward(
            src, R, attention_mask, padding_mask=padding_mask
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
        return output_utterance, output_right_context

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        cache: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass for inference.

         B: batch size;
         D: embedding dimension;
         R: length of right_context;
         U: length of utterance;
         M: length of memory.

        Args:
           utterance (torch.Tensor):
             Utterance frames, with shape (U, B, D).
           right_context (torch.Tensor):
             Right context frames, with shape (R, B, D).
           attn_cache (List[torch.Tensor]):
             Cached attention tensors generated in preceding computation,
             including memory, key and value of left context.
           conv_cache (torch.Tensor, optional):
             Cache tensor of left context for causal convolution.
           padding_mask (torch.Tensor):
             Padding mask of ker tensor.

         Returns:
           (Tensor, Tensor, List[torch.Tensor], Tensor):
             - output utterance, with shape (U, B, D);
             - output right_context, with shape (R, B, D);
             - output attention cache;
             - output convolution cache.
        """
        R = self.right_context_length
        src = torch.cat([right_context, utterance])
        attn_cache = cache[:3]
        conv_cache = cache[3]

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # emformer attention module
        src_att, attn_cache = self._apply_attention_module_infer(
            src, R, attn_cache, padding_mask=padding_mask
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
        return (output_utterance, output_right_context, attn_cache + [conv_cache])


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

    In this model, the memory bank computation is simplifed, using the averaged
    value of each chunk as its memory vector.

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
      memory_size (int, optional):
        Number of memory elements to use. (default: 0)
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
        memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        assert (
            chunk_length - 1
        ) & chunk_length == 0, "chunk_length should be a power of 2."
        self.shift = int(math.log(chunk_length, 2))

        self.use_memory = memory_size > 0

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
                    memory_size=memory_size,
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        self.memory_size = memory_size
        self.cnn_module_kernel = cnn_module_kernel

    def _gen_right_context(self, x: torch.Tensor) -> torch.Tensor:
        """Hard copy each chunk's right context and concat them."""
        T = x.shape[0]
        num_chunks = math.ceil((T - self.right_context_length) / self.chunk_length)
        # first (num_chunks - 1) right context block
        intervals = torch.arange(
            0, self.chunk_length * (num_chunks - 1), self.chunk_length
        )
        first = torch.arange(
            self.chunk_length, self.chunk_length + self.right_context_length
        )
        indexes = intervals.unsqueeze(1) + first.unsqueeze(0)
        # cat last right context block
        indexes = torch.cat(
            [
                indexes,
                torch.arange(T - self.right_context_length, T).unsqueeze(0),
            ]
        )
        right_context_blocks = x[indexes.reshape(-1)]
        return right_context_blocks

    def _gen_attention_mask_col_widths(self, chunk_idx: int, U: int) -> List[int]:
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
            m_start = max(chunk_idx - self.memory_size, 0)
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
        M: length of memory vectors;
        Q: length of attention query;
        KV: length of attention key and value.

        The shape of attention mask is (Q, KV).
        If self.use_memory is `True`:
          query = [right_context, utterance];
          key, value = [memory, right_context, utterance];
          Q = R + U, KV = M + R + U.
        Otherwise:
          query = [right_context, utterance]
          key, value = [right_context, utterance]
          Q = R + U, KV = R + U.

        Suppose:
          c_i: chunk at index i;
          r_i: right context that c_i can use;
          l_i: left context that c_i can use;
          m_i: past memory vectors from previous layer that c_i can use;
        The target chunk-wise attention is:
          c_i, r_i (in query) -> l_i, c_i, r_i, m_i (in key).
        """
        U = utterance.size(0)
        num_chunks = math.ceil(U / self.chunk_length)

        right_context_mask = []
        utterance_mask = []

        if self.use_memory:
            num_cols = 9
            # right context and utterance both attend to memory, right context,
            # utterance
            right_context_utterance_cols_mask = [
                idx in [1, 4, 7] for idx in range(num_cols)
            ]
        else:
            num_cols = 6
            # right context and utterance both attend to right context and
            # utterance
            right_context_utterance_cols_mask = [
                idx in [1, 4] for idx in range(num_cols)
            ]
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

        attention_mask = (
            1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])
        ).to(torch.bool)
        return attention_mask

    def _forward(
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

        right_context = self._gen_right_context(x)
        utterance = x[:U]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        attention_mask = self._gen_attention_mask(utterance)

        M = (
            right_context.size(0) // self.right_context_length - 1
            if self.use_memory
            else 0
        )
        padding_mask = make_pad_mask(M + right_context.size(0) + output_lengths)

        output = utterance
        for layer in self.emformer_layers:
            output, right_context = layer(
                output,
                right_context,
                attention_mask,
                padding_mask=padding_mask,
                warmup=warmup,
            )

        return output, output_lengths

    @torch.jit.export
    def infer(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor],]:
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
          states (List[torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]: # noqa
            Cached states containing:
            - attn_caches: attention states from preceding chunk's computation,
              where each element corresponds to each emformer layer
            - conv_caches: left context for causal convolution, where each
              element corresponds to each layer.

        Returns:
          (Tensor, Tensor, List[List[torch.Tensor]], List[torch.Tensor]):
            - output utterance frames, with shape (U, B, D).
            - output lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
        """
        # lengths = chunk_length + right_context_length
        utterance = x[: self.chunk_length]
        right_context = x[self.chunk_length :]
        #  right_context_utterance = torch.cat([right_context, utterance])

        output = utterance
        output_states: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.emformer_layers):
            start = layer_idx * 4
            end = start + 4
            cache = states[start:end]

            (output, right_context, output_cache,) = layer.infer(
                output,
                right_context,
                padding_mask=None,
                cache=cache,
            )
            output_states.extend(output_cache)

        return output, output_states

    @torch.jit.export
    def init_states(
        self, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        """Create initial states."""
        #
        states = []
        # layer0: attn cache, conv cache, 3 tensors + 1 tensor
        # layer1: attn cache, conv cache, 3 tensors +  1 tensor
        # layer2: attn cache, conv cache, 3 tensors + 1 tensor
        # ...
        # last layer: attn cache, conv cache, 3 tensors + 1 tensor
        for i in range(self.num_encoder_layers):
            states.append(torch.zeros(self.memory_size, 1, self.d_model, device=device))
            states.append(
                torch.zeros(self.left_context_length, 1, self.d_model, device=device)
            )
            states.append(
                torch.zeros(self.left_context_length, 1, self.d_model, device=device)
            )

            states.append(
                torch.zeros(1, self.d_model, self.cnn_module_kernel - 1, device=device)
            )
        return states

        attn_caches = [
            [
                torch.zeros(self.memory_size, self.d_model, device=device),
                torch.zeros(self.left_context_length, self.d_model, device=device),
                torch.zeros(self.left_context_length, self.d_model, device=device),
            ]
            for _ in range(self.num_encoder_layers)
        ]
        conv_caches = [
            torch.zeros(self.d_model, self.cnn_module_kernel - 1, device=device)
            for _ in range(self.num_encoder_layers)
        ]
        states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]] = (
            attn_caches,
            conv_caches,
        )
        return states


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
        memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        is_pnnx: bool = True,
    ):
        super().__init__()

        self.subsampling_factor = subsampling_factor
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        if chunk_length % subsampling_factor != 0:
            raise NotImplementedError(
                "chunk_length must be a mutiple of subsampling_factor."
            )
        if left_context_length != 0 and left_context_length % subsampling_factor != 0:
            raise NotImplementedError(
                "left_context_length must be 0 or a mutiple of subsampling_factor."  # noqa
            )
        if right_context_length != 0 and right_context_length % subsampling_factor != 0:
            raise NotImplementedError(
                "right_context_length must be 0 or a mutiple of subsampling_factor."  # noqa
            )

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model, is_pnnx=is_pnnx)
        self.is_pnnx = is_pnnx

        self.encoder = EmformerEncoder(
            chunk_length=chunk_length // subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
            cnn_module_kernel=cnn_module_kernel,
            left_context_length=left_context_length // subsampling_factor,
            right_context_length=right_context_length // subsampling_factor,
            memory_size=memory_size,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )

    def _forward(
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

        x_lens = (((x_lens - 1) >> 1) - 1) >> 1
        assert x.size(0) == x_lens.max().item()

        output, output_lengths = self.encoder(x, x_lens, warmup=warmup)  # (T, N, C)

        output = output.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return output, output_lengths

    def forward(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor],]:
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
          states (List[torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]: # noqa
            Cached states containing:
            - past_lens: number of past frames for each sample in batch
            - attn_caches: attention states from preceding chunk's computation,
              where each element corresponds to each emformer layer
            - conv_caches: left context for causal convolution, where each
              element corresponds to each layer.
        Returns:
          (Tensor, Tensor):
            - output embedding, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - output lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
        """
        x = self.encoder_embed(x)
        # drop the first and last frames
        x = x[:, 1:-1, :]
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!

        output, output_states = self.encoder.infer(x, states)

        output = output.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return output, output_states

    @torch.jit.export
    def init_states(
        self, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        """Create initial states."""
        return self.encoder.init_states(device)


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
        is_pnnx: bool = False,
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
          is_pnnx:
            True if we are converting the model to PNNX format.
            False otherwise.
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

        # ncnn supports only batch size == 1
        self.is_pnnx = is_pnnx
        self.conv_out_dim = self.out.weight.shape[1]

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

        if torch.jit.is_tracing() and self.is_pnnx:
            x = x.permute(0, 2, 1, 3).reshape(1, -1, self.conv_out_dim)
            x = self.out(x)
        else:
            # Now x is of shape (N, odim, ((T-1)//2-1)//2, ((idim-1)//2-1)//2)
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x
