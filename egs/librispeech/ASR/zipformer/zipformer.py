#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao)
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

import copy
import logging
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
from encoder_interface import EncoderInterface
from scaling import (
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    OrthogonalLinear,
    SimpleOrthogonalLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    ActivationDropoutAndLinear,
    ExpNorm,
    ChunkCausalDepthwiseConv1d,
    CosineSimilarityLoss,
    ScheduledFloat,
    FloatLike,
    SwashR,
    convert_num_channels,
    limit_param_value,
    penalize_abs_values_gt,
    softmax,
    ScaleLimiter,
    with_loss,
)
try:
    from scaling import CorrelationLimiter
except:
    pass


from torch import Tensor, nn


class Zipformer2(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_multiple (int or Tuple[int]): determines hidden dimension in feedforward modules
        conv_params (int or Tuple[int])): Kernel size of convolution module

        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.
    """
    def __init__(
        self,
        input_dim: int,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        query_head_dim: Union[int, Tuple[int]] = 64,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_multiple: Union[int, Tuple[int]] = 4,
        conv_params: Union[int, Tuple[int]] = 31,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super(Zipformer2, self).__init__()

        def _to_tuple(x):
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor  # int

        self.downsampling_factor = downsampling_factor  # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_multiple = _to_tuple(feedforward_multiple)
        self.conv_params = conv_params = _to_tuple(conv_params)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        # each one will be Zipformer2Encoder or OrthogonalDownsample or OrthogonalUpsample
        encoders = []

        num_encoders = len(downsampling_factor)

        # caution: some changes we made for this break the streaming, later we'll try to fix this.
        encoders_downsampling_factors = [ ]

        # make it so large the limit is never reached.
        max_proj_dim = max(downsampling_factor) * max(encoder_dim)


        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_multiple=feedforward_multiple[i],
                conv_params=conv_params[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                dim=downsampling_factor[i]*input_dim,
            )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

    def get_chunk_info(self) -> Tuple[int, int]:
        """
         Returns chunk_size and left_context_chunks.
         """
        if not self.causal:
            return -1, -1

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert len(self.chunk_size) == 1, self.chunk_size
            chunk_size = self.chunk_size[0]
        else:
            chunk_size = random.choice(self.chunk_size)

        if chunk_size == -1:
            left_context_chunks = -1
        else:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                assert len(self.left_context_frames) == 1, self.left_context_frames
                left_context_frames = self.left_context_frames[0]
            else:
                left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1

        return chunk_size, left_context_chunks

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
          aux_loss_scale:
            If supplied, auxiliary losses such as CosineSimilarityLoss will be
            applied with this scale on the loss (note, these aux losses are
            reduced via summation over frames.)
        Returns:
          Return (embeddings_lengths), where:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        chunk_size, left_context_chunks = self.get_chunk_info()
        orig_seq_len = x.shape[0]

        pad = (-orig_seq_len) % max(self.downsampling_factor)
        # pad sequence length to be multiple of max(self.downsampling_factor)
        x = torch.cat((x, x[-1:].repeat(pad, 1, 1)),
                      dim=0)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # Not support exporting a model for simulating streaming decoding
            attn_mask = None
        else:
            attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        src_key_padding_mask = pad_mask(src_key_padding_mask, x.shape[0])

        num_stacks = len(self.downsampling_factor)


        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = downsample_by(x, ds)
            T = x.shape[0]
            x = module(
                x,
                chunk_size=chunk_size,
                src_key_padding_mask=(
                    None
                    if src_key_padding_mask is None
                    else src_key_padding_mask[..., ::ds]
                ),
                attn_mask=(None
                           if attn_mask is None
                           else attn_mask[::ds, ::ds]
                ),
                aux_loss_scale=aux_loss_scale * ds / (self.output_downsampling_factor * num_stacks)
            )
            x = upsample_by(x, ds)

        od = self.output_downsampling_factor
        x = downsample_by(x, od)
        x = x[:(orig_seq_len + od - 1) // od]  # truncate so seq len not affected by padding

        if od > 1:
            x_lens = (x_lens + od - 1) // od

        return x, x_lens

    def _get_attn_mask(
        self, x: Tensor, chunk_size: int, left_context_chunks: int
    ) -> Optional[Tensor]:
        """
        Return None if chunk_size == -1, else return attention mask of shape
          (seq_len, seq_len), interpreted as (tgt_seq_len, src_seq_len).  True
           means a masked position.
        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
          chunk_size: chunk size, must divide
        """
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all(
                chunk_size * left_context_chunks
                >= (self.conv_params[i] // 2) * self.downsampling_factor[i]
                for i in range(num_encoders)
            )
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            c = t // chunk_size
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c, src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask


    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states: list of cached tensors of all encoder layers. For layer-i,
            states[i*5:(i+1)*5] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
            cached_conv)
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
            - updated states
        """
        new_states = []
        layer_offset = 0

        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            ds = self.downsampling_factor[i]

            x, new_layer_states = module.streaming_forward(
                x,
                states=states[layer_offset * 6 : (layer_offset + num_layers) * 5],
                left_context_len=self.left_context_frames[0] // ds,
                src_key_padding_mask=src_key_padding_mask[..., ::ds],
            )
            layer_offset += num_layers
            new_states += new_layer_states

        x = x[..., :max(self.encoder_dim)]  # for historical reasons.  can change this.

        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            lengths = (x_lens + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lens + 1) // 2

        return x, lengths, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Get initial states.

        A list of cached tensors of all encoder layers. For layer-i, states[i*5:(i+1)*5]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        """
        states = []
        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            embed_dim = self.encoder_dim[i]
            ds = self.downsampling_factor[i]
            num_heads = self.num_heads[i]
            key_dim = self.query_head_dim[i] * num_heads
            value_dim = self.value_head_dim[i] * num_heads
            downsample_left = self.left_context_frames[0] // ds
            nonlin_attn_head_dim = 3 * embed_dim // 4
            conv_left_pad = self.cnn_module_kernel[i] // 2  # will be error. have to figure this out.
            for layer in range(num_layers):
                cached_key = torch.zeros(downsample_left, batch_size, key_dim).to(
                    device
                )
                cached_nonlin_attn = torch.zeros(
                    1, batch_size, downsample_left, nonlin_attn_head_dim
                ).to(device)
                cached_val1 = torch.zeros(downsample_left, batch_size, value_dim).to(
                    device
                )
                cached_val2 = torch.zeros(downsample_left, batch_size, value_dim).to(
                    device
                )
                cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(
                    device
                )
                cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(
                    device
                )
                states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]

        return states


def pad_mask(mask: Optional[Tensor], seq_len: int):
    # mask: (batch_size, old_seq_len)
    # if mask is not None, returns mask: (batch_size, seq_len); pads with True (i.e., masked).
    if mask is None:
        return None
    (batch_size, old_seq_len) = mask.shape
    pad = seq_len - old_seq_len
    if pad == 0:
        return mask
    else:
        return torch.cat((mask, torch.ones(batch_size, pad, device=mask.device, dtype=torch.bool)),
                         dim=1)


def downsample_by(x: Tensor, downsampling_factor: int) -> Tensor:
    # x: (seq_len, batch_size, num_channels)
    # Returns: (seq_len // downsampling_factor, batch_size, num_channels * downsampling_factor)
    if downsampling_factor == 1:
        return x
    (seq_len, batch_size, num_channels) = x.shape
    x = x.reshape(seq_len // downsampling_factor, downsampling_factor, batch_size, num_channels)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(seq_len // downsampling_factor, batch_size, downsampling_factor * num_channels)
    return x

def upsample_by(x: Tensor, upsampling_factor: int) -> Tensor:
    # x: (seq_len, batch_size, num_channels)
    # Returns: (seq_len * upsampling_factor, batch_size, num_channels // upsampling_factor)
    if upsampling_factor == 1:
        return x
    (seq_len, batch_size, num_channels) = x.shape
    x = x.reshape(seq_len, batch_size, upsampling_factor, num_channels // upsampling_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(seq_len * upsampling_factor, batch_size, num_channels // upsampling_factor)
    return x


def get_dct_matrix(N):
    """
    Generates an orthonormal DCT-II matrix for a given size N.
    Args:
        N (int): The size of the square matrix.
    Returns:
        torch.Tensor: The N x N orthonormal DCT-II matrix.
    """
    # Create the base matrix with dimensions (N, N)
    mat = torch.zeros(N, N)
    # Create a tensor for the indices k (rows) and n (columns)
    k = torch.arange(N).unsqueeze(1)
    n = torch.arange(N).unsqueeze(0)
    # Fill the matrix using the DCT-II formula
    mat = math.sqrt(2 / N) * torch.cos(math.pi / (2 * N) * (2 * n + 1) * k)
    # Adjust the first row (k=0) with a special normalization factor
    mat[0] *= (2 ** -0.5)
    return mat


class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_multiple: determines the hidden dimension of the feedforward module

        conv_params (int): params per channel of convolution module

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query_head_dim: int,
        value_head_dim: int,
        feedforward_multiple: int,
        conv_params: int,
        causal: bool = False,
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.name = None  # will be set from training loop

        self.residual_scale = nn.Parameter(0.5 * torch.ones(embed_dim))

        self.offset_scale_limiter = ScaleLimiter(max_rms=0.5)
        self.offset_correlation_limiter = CorrelationLimiter()

        self.self_attn_weights = MultiheadAttentionWeights(
            embed_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
        )

        self.self_attn = GatedSelfAttention(embed_dim, num_heads, value_head_dim)

        feedforward_dim = embed_dim * feedforward_multiple
        self.feed_forward1 = FeedforwardModule(embed_dim, feedforward_dim)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim)

        self.conv_module = ConvolutionModule(embed_dim, conv_params, causal=causal)

        self.norm = ExpNorm(embed_dim)


    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tensor:
        """
            Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
             chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
             attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                    interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                   True means masked position. May be None.
        src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.
          aux_loss_scale:
            If supplied, auxiliary losses such as CosineSimilarityLoss will be
            applied with this scale on the loss (note, these aux losses are
            reduced via summation over frames.)

            Returns:
               A tensor which has the same shape as src
        """
        src_orig = src

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            aux_loss_scale=0.1 * aux_loss_scale,
        )

        src = src + self.feed_forward1(src, aux_loss_scale=0.1 * aux_loss_scale, src_key_padding_mask=src_key_padding_mask)

        src = src + self.self_attn(src, attn_weights, aux_loss_scale=0.1 * aux_loss_scale, src_key_padding_mask=src_key_padding_mask)

        src = src + self.conv_module(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask, aux_loss_scale=0.1 * aux_loss_scale)

        src = src + self.feed_forward2(src, aux_loss_scale=0.1 * aux_loss_scale, src_key_padding_mask=src_key_padding_mask)

        residual_scale = limit_param_value(self.residual_scale, min=0.25, max=0.75)
        offset = (src - src_orig) * residual_scale

        offset = self.offset_scale_limiter(offset, aux_loss_scale)

        offset = with_loss(offset,
                           self.offset_correlation_limiter(
                               offset.permute(1, 0, 2), src_orig.permute(1, 0, 2),
                               aux_loss_scale, mask=src_key_padding_mask))

        src = src_orig + offset

        src = self.norm(src)

        return src

    def streaming_forward(
        self,
        src: Tensor,
        cached_key: Tensor,
        cached_nonlin_attn: Tensor,
        cached_val1: Tensor,
        cached_val2: Tensor,
        cached_conv: Tensor,
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pass the input through the encoder layer in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            cached_nonlin_attn: left context for nonlin_attention module, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            cached_val1: cached left context for the first attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_val2: cached left context for the second attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_conv: cached left context for the first convolution module,
              of shape (batch_size, channels, left_pad)
            left_context_len: number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
            - x, with the same shape as src
            - updated cached_key
            - updated cached_nonlin_attn
            - updated cached_val1
            - updated cached_val2
            - updated cached_conv
        """
        src_orig = src

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights, cached_key = self.self_attn_weights.streaming_forward(
            src,
            cached_key=cached_key,
            left_context_len=left_context_len,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)


        na, cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src,
            attn_weights[0:1],
            cached_x=cached_nonlin_attn,
            left_context_len=left_context_len,
        )
        src = src + na

        self_attn, cached_val1 = self.self_attn1.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val1,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv = self.conv_module.streaming_forward(
            src,
            cache=cached_conv,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)


        self_attn, cached_val2 = self.self_attn2.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val2,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        offset = (src - src_orig) * self.residual_scale

        src = src_orig + offset

        src = self.norm(src)

        return (
            src,
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv,
        )


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
         dim:  the dimension of the input and output (layer dim may be less than this).

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dim: int,
    ) -> None:
        super().__init__()

        # self.downsample will also reverse the downsampling operation for us afterward.
        self.proj = SimpleOrthogonalLinear(dim, encoder_layer.embed_dim, bias=False)
        self.proj.lr_scale = 0.75

        self.name = None
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        self.residual_scales = nn.Parameter(
            torch.cat([ -1.0 * torch.ones(1),
                        (1. / num_layers) * torch.ones(num_layers) ],
                      dim=0))


        self.copy_bypass = Identity()



    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embed_dim),
                 but embed_dim is allowed to exceed the modules' embed_dim; we will bypass
                 any extra dimensions.
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns:
             (out, out_sd), both of the same shape as src,
           where out_sd is an alternative version of out for stochastic-depth, that does not see the bypass.
        """
        src_orig_fulldim = src

        src = self.proj(src)  # project to layer dim.

        num_layers = len(self.layers)
        src_orig = src

        residual_scale = limit_param_value(self.residual_scales[0],
                                           min=-1.0, max=-0.5)

        src_with_bypass = residual_scale * src

        for i, mod in enumerate(self.layers):
            src = mod(
                src,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
                aux_loss_scale=aux_loss_scale/num_layers,
            )
            residual_scale = limit_param_value(self.residual_scales[i + 1],
                                               min=0.0 if i + 1 < num_layers else 0.25,
                                               max=1.0)
            src_with_bypass = src_with_bypass + residual_scale * src


        offset = src_with_bypass

        src = src_orig_fulldim + self.proj(offset, transpose=True)
        # in effect src_orig_fulldim already contains src_orig with a scale of 1 for the missing dims,
        # because of some identities involving orthogonal matrices.


        return src


    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embed_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*5:(i+1)*5] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv).
            left_context_len: Number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
          - output, a Tensor with the same shape as src.
          - updated states
        """
        num_channels = src.shape[-1]
        layer_dim = self.layers[0].embed_dim
        if num_channels > layer_dim:
            src, bypass = src[..., :layer_dim], src[..., layer_dim:]

        new_states = []
        for i, mod in enumerate(self.layers):
            (
                cached_key,
                cached_nonlin_attn,
                cached_val1,
                cached_val2,
                cached_conv,
            ) = states[i * 5 : (i + 1) * 5]
            (
                src,
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv,
            ) = mod.streaming_forward(
                src,
                cached_key=cached_key,
                cached_nonlin_attn=cached_nonlin_attn,
                cached_val1=cached_val1,
                cached_val2=cached_val2,
                cached_conv=cached_conv,
                left_context_len=left_context_len,
                src_key_padding_mask=src_key_padding_mask,
            )
            new_states += [
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv,
            ]

        if num_channels > layer_dim:
            src = torch.cat((src, bypass), dim=-1)

        return src, new_states


class ResidualModule(nn.Module):
    """
    An nn.Module that implements a learnable residual scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
            self,
            embed_dim: int,
            function_scale_min: FloatLike = 0.1,
    ):
        super().__init__()
        self.function_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.function_scale_min = copy.deepcopy(function_scale_min)


    def _get_scales(self):
        function_scale = self.function_scale
        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and self.training:
            function_scale = limit_param_value(
                function_scale, min=float(self.function_scale_min), max=1.0,
            )
        residual_scale = 1.0 - function_scale
        return residual_scale, function_scale

    def forward(self, src_orig: Tensor, src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        residual_scale, function_scale = self._get_scales()
        return residual_scale * src_orig + function_scale * src



# taken from torchtune.
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
     """
    def __init__(
            self,
            dim: int,
            max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        assert dim in [64, 128]
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        # these multiples are on the inverse frequences, so on frequencies the multiples would be the inverses of these.
        # it's the frequencies we want to be exact multiples of each other.
        if self.dim == 64:
            multiples = [ 1., 4. / 3. ]
        else:
            assert self.dim == 128
            multiples = [ 1., 4. / 3., 8. / 5., 8. / 7. ]
        assert self.dim % (2 * len(multiples)) == 0   # e.g. self.dim == 128.  head dim.
        D = self.dim // (2 * len(multiples))          # e.g. D == 16.

        inv_freqs = (2. ** torch.arange(D))  # [ 1, 2, 4, ... ]
        inv_freqs = torch.cat([ m * inv_freqs for m in multiples ], dim=0)

        angular_freqs = math.pi / inv_freqs
        # so highest angular_freq is pi, which means flipping between -1 and 1 on alternate tokens.  this is
        # the nyquist.
        self.register_buffer("theta", angular_freqs, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class MultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with additive relative-position
    scores that are kept separate from the regular scores.

    relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.

    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query_head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.

        self.attn_score_limit = ScheduledFloat((0.0, 5.0), (5000.0, 20.0))
        self.attn_score_penalty_prob = ScheduledFloat((0.0, 1.0), (5000.0, 1.0), (5001.0, 0.1))

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim,
            bias=True, initial_scale=0.125 * query_head_dim**-0.25
        )

        self.rope = RotaryPositionalEmbeddings(query_head_dim) # use default max_seq_len=4096

        self.copy_query = Identity()

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]

        q = self.copy_query(q)  # for diagnostics only, does nothing.

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        q = self.rope(q.permute(1, 0, 2, 3))  # (batch, seq, head, channel)
        k = self.rope(k.permute(1, 0, 2, 3))  # (batch, seq, head, channel)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 0, 1, 3)  # (head, batch, time1, query_head_dim)
        k = k.permute(2, 0, 3, 1)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)


        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size,
                seq_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and self.training:
            attn_scores_limit = 8.0  # limit on our metric that affects how much grad we are likely to backpropagate.
            attn_scores = PenalizeLargeAttentionScores.apply(attn_scores, attn_scores_limit, 0.1 * aux_loss_scale,
                                                             key_padding_mask, self.name)



        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights

    def streaming_forward(
        self,
        x: Tensor,
        cached_key: Tensor,
        left_context_len: int,
        key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            left_context_len: number of left context frames.
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
              are True in this mask will be ignored as sources in the attention weighting.

        Returns:
           - attention weights, of shape (hum_heads, batch_size, seq_len, seq_len2),
             interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
           - updated cached attention key tensor of left context.
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]

        # Pad cached left contexts
        assert cached_key.shape[0] == left_context_len, (
            cached_key.shape[0],
            left_context_len,
        )
        k = torch.cat([cached_key, k], dim=0)
        # Update cached left contexts
        cached_key = k[-left_context_len:, ...]

        # The length of key
        k_len = k.shape[0]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        k = k.reshape(k_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        assert attn_scores.shape == (
            num_heads,
            batch_size,
            seq_len,
            k_len,
        ), attn_scores.shape

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, k_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )


        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and self.training:
            attn_scores_limit = 8.0  # limit on our metric that affects how much grad we are likely to backpropagate.
            attn_scores = PenalizeLargeAttentionScores.apply(attn_scores, attn_scores_limit, 0.1 * aux_loss_scale,
                                                             key_padding_mask, self.name)

        attn_weights = attn_scores.softmax(dim=-1)

        return attn_weights, cached_key

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .mean(dim=(1, 2))
                )
                logging.info(
                    f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}"
                )



class PenalizeLargeAttentionScores(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            attn_scores: Tensor,
            limit: float,
            aux_loss_scale: float,
            key_padding_mask: Optional[Tensor],
            name: str):
        # attn_scores: (head, batch, query_time, key_time)
        ctx.save_for_backward(attn_scores)
        ctx.mask = key_padding_mask    # has no grad
        ctx.limit = limit
        ctx.aux_loss_scale = aux_loss_scale
        ctx.name = name
        return attn_scores

    @staticmethod
    def backward(
            ctx,
            attn_scores_grad):
        attn_scores, = ctx.saved_tensors
        mask = ctx.mask
        (num_heads, batch_size, seq_len, _) = attn_scores.shape
        with torch.amp.autocast('cuda', enabled=False):
            attn_scores = attn_scores.to(torch.float)
            attn_scores = attn_scores.detach()
            # attn_scores: (head, batch, query_time, key_time)
            attn_scores.requires_grad = True
            with torch.enable_grad():
                probs = attn_scores.softmax(dim=-1)
                scaled_scores = attn_scores.abs() * probs
                avg_scores = scaled_scores.sum(dim=-1) # (head, batch, query_time)
                if mask is not None:
                    avg_scores = avg_scores * (~mask) # mask: (batch, time)
                query_scores = (avg_scores - ctx.limit).relu()

                if random.random() < 0.0005:
                    query_excess = query_scores.mean(dim=(1,2)).to('cpu')
                    avg_scores_mean = avg_scores.mean(dim=(1,2)).to('cpu')
                    logging.info(f"PenalizeLargeAttentionScores: {ctx.name}, limit={ctx.limit}, avg_scores={avg_scores_mean}, query_excess={query_excess}")
                # all these losses have a "per-frame" scaling, i.e. scaled proportional to the total number
                # of frames which is batch_size * seq_len.  normalize by dividing by num heads.
                # also divide by ctx.limit so it's like penalizing a relative excess.
                query_scores.backward(gradient=torch.full_like(query_scores, ctx.aux_loss_scale / (num_heads * ctx.limit)))

        return attn_scores_grad + attn_scores.grad, None, None, None, None





class GatedSelfAttention(nn.Module):
    """
    Self-attention module with sigmoid gating.  This one works with already-computed attention
    weights, e.g. as computed by MultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = ScaledLinear(embed_dim, 2 *num_heads * value_head_dim,
                                    bias=True)

        self.sigmoid = nn.Sigmoid()

        self.out_proj = ScaledLinear(
             num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

        f = max(1.0, embed_dim / (num_heads * value_head_dim))



    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        aux_loss_scale: float = 0.0,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
         src_key_padding_mask: optional Tensor of shape (batch_size, src_seq_len); only
          used for the cosine similarity loss, during training.
        Returns:
           a tensor with the same shape as x.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)  # (seq_len, batch_size, 2 * num_heads * value_head_dim)
        x, s = x.chunk(2, dim=-1)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # x: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = x * self.sigmoid(s)
        x = self.out_proj(x)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: input tensor, of shape (seq_len, batch_size, embed_dim)
            attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
              with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
              attn_weights.sum(dim=-1) == 1.
            cached_val: cached attention value tensor of left context,
              of shape (left_context_len, batch_size, value_dim)
            left_context_len: number of left context frames.

        Returns:
           - attention weighted output, a tensor with the same shape as x.
           - updated cached attention value tensor of left context.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        seq_len2 = seq_len + left_context_len
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len2)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)

        # Pad cached left contexts
        assert cached_val.shape[0] == left_context_len, (
            cached_val.shape[0],
            left_context_len,
        )
        x = torch.cat([cached_val, x], dim=0)
        # Update cached left contexts
        cached_val = x[-left_context_len:, ...]

        x = x.reshape(seq_len2, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)

        return x, cached_val


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(self, embed_dim: int, feedforward_dim: int):
        super(FeedforwardModule, self).__init__()
        # try to get in the useful range of the activation function, i.e. not too small.
        self.in_proj = ScaledLinear(embed_dim, feedforward_dim)
        # weight_min_rms will be interpreted by get_parameter_groups_with_lrs() and passed
        # to the TransformedAdam optimizer.
        self.in_proj.weight_min_rms = 0.02

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            dropout_p=0.0,
            activation="SwashR",
            initial_scale=0.5,
            bias=True,
        )


    def forward(self, x: Tensor, aux_loss_scale: float = 0.0, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.in_proj(x)
        x = self.out_proj(x)
        return x

def round_up_to_power_of_two(x):
    x = x - 1
    x = x | x >> 1
    x = x | x >> 2
    x = x | x >> 4
    x = x | x >> 8
    x = x | x >> 16
    x = x + 1
    return x




class FftConv(nn.Module):
    def __init__(self,
                 num_channels: int,
                 params_per_channel: int,
                 bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_channels, params_per_channel))
        # one factor of 2 is for (sin, cos); the other is to double the num representable freqs
        self.weight_proj = nn.Linear(params_per_channel, 4 * params_per_channel)

        if bias:
            self.bias = nn.Parameter(0.01 * torch.randn(num_channels))


    def forward(self,
                x: Tensor) -> Tensor:
        (seq_len, batch_size, num_channels) = x.shape


        with torch.amp.autocast('cuda', enabled=False):
            # do it in float32 because non power of two seq_len is not supported in half precision.
            x = torch.fft.rfft(x.to(torch.float32), dim=0)
            # x: (num_freqs, batch_size, num_channels)
            N = x.shape[0]   # num freqs
            weight = self.weight_proj(self.weight).reshape(num_channels, 2, -1)
            weight = torch.nn.functional.interpolate(weight, N, mode='linear', align_corners=True)
            weight = torch.view_as_complex(weight.permute(2, 0, 1).contiguous())
            # weight: (N, num_channels)
            weight = weight.unsqueeze(1)  # (N, 1, num_channels)
            x = x * weight
            x = torch.fft.irfft(x, n=seq_len, dim=0)

        try:
            x = x + self.bias
        except AttributeError:
            pass

        return x




class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            3 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.


        self.activation1 = Identity()  # for diagnostics

        self.sigmoid1 = nn.Sigmoid()

        self.sigmoid2 = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        self.depthwise_conv = FftConv(bottleneck_dim, kernel_size)

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwashR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
        aux_loss_scale: float = 0.0,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        # x: (time, batch, channels)
        # Caution: this module is not completely
        # invariant to the number of frames each sequence is padded with, since
        # the FFT-based convolution treats the signal as repeating.
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.t().unsqueeze(-1).expand_as(x), 0.0)

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s, y = x.chunk(3, dim=2)
        s = self.sigmoid1(s)
        y = self.sigmoid2(y)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        x = self.depthwise_conv(x)   # x: (time, batch, bottleneck_dim)

        x = x * y
        x = self.out_proj(x)  # (time, batch, channels)

        return x


    def repeat_in_padding(self, x, mask):
        # repeats elements of x in the padding region, circularly as much as possible;
        # the discontinuity between the ones that circularly repeat from the end and
        # those that circularly repeat from the beginning is in the middle of the padding
        # region.

        # x: (seq_len, batch_size, num_channels)
        (batch_size, seq_len) = mask.shape

        seq_lengths = (~mask).to(torch.int64).sum(dim=1, keepdim=True)  # (batch_size, 1)
        pad_len = seq_len - seq_lengths
        arange = torch.arange(seq_len, device=mask.device)

        # "mid" gives the index of the midpoint of the padding region after each sequence.
        mid = (seq_lengths + seq_len) // 2  # mid: (batch_size, 1)

        src_index = torch.where(arange >= mid, arange - pad_len, arange) % seq_lengths
        # src_index: (batch_size, seq_len)

        src_index = src_index.t().unsqueeze(-1).expand_as(x)
        # src_index: (seq_len, batch_size, num_channels)
        x = torch.gather(x, dim=0, index=src_index)
        return x



    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module in streaming forward mode.

        Args:
            x: Input tensor (#time, batch, channels).
            cache: cached left context for depthwise_conv of shape
              (#batch, channels, left_pad)
            src_key_padding_mask: the mask for the src keys per batch (optional):
              (batch, #time), contains True in masked positions.

        Returns:
            - Output tensor (#time, batch, channels).
            - Updated cache (#batch, channels, left_pad)
        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        x, cache = self.depthwise_conv.streaming_forward(x, cache=cache)

        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.out_proj(x)  # (time, batch, channels)

        return x, cache


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def _test_zipformer_main(causal: bool = False):
    seq_len = 20
    # Just make sure the forward pass runs.

    input_dim = 50

    c = Zipformer2(
        input_dim=input_dim,
        encoder_dim=(64, 96),
        num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
    )

    batch_size = 6
    seq_len = 21
    # Just make sure the forward pass runs.
    f, lengths = c(
        torch.randn(seq_len, batch_size, input_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        aux_loss_scale=1.0,
    )
    f.sum().backward()
    c.eval()
    x_ = c(
        torch.randn(seq_len, batch_size, input_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        aux_loss_scale=1.0,
    )
    x_  # to remove flake8 warnings



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
