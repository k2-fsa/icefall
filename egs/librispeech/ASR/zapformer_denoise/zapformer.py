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
from scaling import (
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    OrthogonalLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    ScaleLimiter,
    ActivationDropoutAndLinear,
    ExpNorm,
    ChunkCausalDepthwiseConv1d,
    Dropout2,
    FloatLike,
    ScheduledFloat,
    Whiten,
    convert_num_channels,
    limit_param_value,
    penalize_abs_values_gt,
    softmax,
)
from torch import Tensor, nn


class Zapformer(nn.Module):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
       time_embed_dim: an integer giving the dimension of the time embeddings provided
          to the network.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_multiple (int or Tuple[int]): determines hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
    """
    def __init__(
        self,
        input_dim: int,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        time_embed_dim: int = 256,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_multiple: Union[int, Tuple[int]] = 4,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super().__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

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


        self.downsampling_factor = downsampling_factor  # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_multiple = _to_tuple(feedforward_multiple)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        # each one will be ZapformerEncoder or OrthogonalDownsample or OrthogonalUpsample
        encoders = []

        num_encoders = len(downsampling_factor)
        cur_downsample = 1

        # caution: some changes we made for this break the streaming, later we'll try to fix this.
        encoders_downsampling_factors = [ ]

        # make it so large the limit is never reached.
        max_proj_dim = max(downsampling_factor) * max(encoder_dim)

        def set_downsample_factor(cur_downsample, ds):
            while cur_downsample < ds:
                # need to downsample
                encoders.append(OrthogonalDownsample(channels=input_dim * cur_downsample,
                                                     proj_dim=min(2 * input_dim * cur_downsample, max_proj_dim)))
                cur_downsample *= 2
            while cur_downsample > ds:
                encoders.append(OrthogonalUpsample(channels=input_dim * cur_downsample,
                                                   proj_dim=min(input_dim * cur_downsample, max_proj_dim)))
                cur_downsample //= 2
            return cur_downsample

        for i in range(num_encoders):
            cur_downsample = set_downsample_factor(cur_downsample, downsampling_factor[i])

            encoder_layer = ZapformerEncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_multiple=feedforward_multiple[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = ZapformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                dim=cur_downsample*input_dim,
                pos_dim=pos_dim,
                time_embed_dim=time_embed_dim,
            )
            encoder.encoder_index = i
            encoders.append(encoder)

        cur_downsample = set_downsample_factor(cur_downsample, 1)

        self.encoders = nn.ModuleList(encoders)


    def forward(
        self,
        x: Tensor,
        time_embed: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, encoder_dim).
          time_embed:
            The timestep-embedding tensor.  Its shape is (batch_size, time_embed_dim)
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return embeddings with the same shape as x: (seq_len, batch_size, encoder_dim)
        """
        orig_seq_len = x.shape[0]

        def truncate(x, downsampling_factor):
            max_len = (orig_seq_len + downsampling_factor - 1) // downsampling_factor
            return x[:max_len] if x.shape[0] > max_len else x


        for module in self.encoders:
            if isinstance(module, ZapformerEncoder):
                i = module.encoder_index  # was set in this class's __init__ function.
                ds = self.downsampling_factor[i]
                x = truncate(x, ds)
                x = module(
                    x,
                    time_embed,
                    src_key_padding_mask=(
                        None
                        if src_key_padding_mask is None
                        else src_key_padding_mask[..., ::ds]
                    ),
                )
            else:
                x = module(x)

        x = x[:orig_seq_len]
        return x



def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))

class ZapformerEncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_multiple: determines the hidden dimension of the feedforward module
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module (default=31).

    Examples::
        >>> encoder_layer = ZapformerEncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """
    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_multiple: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        randomize_scale: FloatLike = ScheduledFloat((0.0, 1.0), (20000.0, 0.75)),
    ) -> None:
        super(ZapformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.name = None  # will be set from training loop

        self.randomize_scale = copy.deepcopy(randomize_scale)
        # self.bypass implements layer skipping as well as learnable scale on a residual term; see its default values.
        self.residual = ResidualModule(
            embed_dim,
        )

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1, self.self_attn2 = [ SelfAttention(embed_dim, num_heads, value_head_dim) for _ in range(2) ]

        feedforward_dim = embed_dim * feedforward_multiple
        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, dropout)

        self.conv_module1, self.conv_module2 = [ ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)
                                                 for _ in range(2) ]

        self.scale_limiter = ScaleLimiter(max_var=2.0)

        self.norm = ExpNorm(embed_dim)


    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
             pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
             chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
             attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                    interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                   True means masked position. May be None.
        src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

            Returns:
               A tensor which has the same shape as src
        """
        src_orig = src

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        src = src + self.self_attn1(src, attn_weights)

        src = src + self.conv_module1(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)

        src = src + self.feed_forward2(src)

        src = src + self.self_attn2(src, attn_weights)

        src = src + self.conv_module2(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)

        src = src + self.feed_forward3(src)

        src = self.residual(src_orig, src)

        src = self.scale_limiter(src)

        src = self.norm(src)

        return src


class ZapformerEncoder(nn.Module):
    r"""ZapformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ZapformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
         dim:  the dimension of the input and output (layer dim may be less than this).
       pos_dim: the dimension for the relative positional encoding
dropout:

    Examples::
        >>> encoder_layer = ZapformerEncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = ZapformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)


    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dim: int,
        pos_dim: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(
            pos_dim, dropout_rate=0.0, length_factor=1.0
        )
        self.name = None
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        self.residual = ResidualModule(encoder_layer.embed_dim)

        self.time_embed = ScaledLinear(time_embed_dim, encoder_layer.embed_dim, initial_scale=0.1)

        #bypass_dim = dim - encoder_layer.embed_dim
        self.copy_bypass = Identity()

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(3.0),
            prob=(1, 1),
            grad_scale=0.025,
        )



    def forward(
        self,
        src: Tensor,
        time_embed: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embed_dim),
                 but embed_dim is allowed to exceed the modules' embed_dim; we will bypass
                 any extra dimensions.
           time_embed: the time embedding, shape: (batch_size, seq_len)
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)

        num_channels = src.shape[-1]
        layer_dim = self.layers[0].embed_dim
        if num_channels > layer_dim:
            src, bypass = src[..., :layer_dim], src[..., layer_dim:]


        src_orig = src
        src = src + self.time_embed(time_embed)
        for i, mod in enumerate(self.layers):
            src = mod(
                src,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            # randomize_factor can be viewed as a simple version of an
            # importance-sampling factor.

        src = self.residual(src_orig, src)
        src = self.whiten(src)

        if num_channels > layer_dim:
            bypass = self.copy_bypass(bypass)
            src = torch.cat((src, bypass), dim=-1)

        return src


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



class OrthogonalDownsample(torch.nn.Module):
    """
    Does downsampling with an orthogonal matrix, by a factor of two.  Projection is initialized
    in a special way and enforced to be orthogonal.

    Args:
       channels: the number of input channels; the num output channels will be twice this
       proj_dim: the number of channels, after combining 2 frames by interpolating their channels
                  as [ a b a b, .. ] that will actually be projected; the rest are just copied.
                  proj_dim=2 * channels would mean all channels are projected in a learned way
         causal: True for causal systems, only affects error messages as requires even
                 input num frames.
    """
    def __init__(
            self, channels: int, proj_dim: int, causal: bool = False,
    ):
        super().__init__()
        assert proj_dim <= channels * 2
        self.proj = OrthogonalLinear(proj_dim, proj_dim, bias=False)
        # lr_scale is a learning-rate factor to slow down how fast self.proj is learned.
        # it will be interpreted by get_parameter_groups_with_lrs()
        self.proj.lr_scale = 0.75
        self.causal = causal

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape

        if seq_len % 2 == 1:
            if torch.jit.is_tracing():
                assert (
                    not self.causal
                ), f"pad should be zero for exporting streaming models. Given {pad}"
            src = torch.cat((src, src[-1:]), dim=0)
            seq_len += 1

        # the following will place each 2 frames of a particular channel right after
        # each other as if they were two different channels.
        src = torch.stack((src[0::2], src[1::2]), dim=-1)
        src = src.reshape(seq_len // 2, batch_size, in_channels * 2)
        proj_channels = self.proj.weight.shape[0]
        if proj_channels < in_channels * 2:
            src = torch.cat((self.proj(src[..., :proj_channels]), src[..., proj_channels:]),
                            dim=-1)
        else:
            src = self.proj(src)
        return src

class OrthogonalUpsample(torch.nn.Module):
    """
    A very simple form of upsampling with an orthogonal matrix.

       proj_dim: the number of channels that will actually be projected; the rest are just copied.
                  proj_dim=channels would mean all channels are projected in a learned way

    """
    def __init__(self, channels: int, proj_dim: int):
        super().__init__()
        assert proj_dim <= channels
        # gradually make smaller and then turn off the non-orthognality penalty.
        self.proj = OrthogonalLinear(proj_dim, proj_dim, bias=False,
                                     penalty_scale=ScheduledFloat((0.0, 20.0), (5000.0, 1.0), (10000.0, 0.1), (20000.0, 0.0)))
        # lr_scale is a learning-rate factor to slow down how fast self.proj is learned.
        # it will be interpreted by get_parameter_groups_with_lrs()
        self.proj.lr_scale = 0.75


    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*2), batch_size, num_channels // 2)
        """
        proj_channels = self.proj.weight.shape[0]
        (seq_len, batch_size, in_channels) = src.shape

        if proj_channels < in_channels:
            src = torch.cat((self.proj(src[..., :proj_channels]), src[..., proj_channels:]),
                            dim=-1)
        else:
            src = self.proj(src)

        src = torch.stack((src[..., 0::2], src[..., 1::2]),
                          dim=1)  # (seq_len, 2, batch_size, in_channels // 2)
        src = src.reshape(seq_len * 2, batch_size, in_channels // 2)
        return src


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the Fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish individual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embed_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, embed_dim
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0, length_factor
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        """Reset the positional encodings."""
        T = x.size(0) + left_context_len

        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= T * 2 - 1:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        """Create positional encoding.

        Args:
            x (Tensor): Input tensor (time, batch, `*`).
            left_context_len: (int): Length of cached left context.

        Returns:
            positional embedding, of shape (batch, left_context_len + 2*time-1, `*`).
        """
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        # length of positive side: x.size(0) + left_context_len
        # length of negative side: x.size(0)
        pos_emb = self.pe[
            self.pe.size(0) // 2
            - x_size_left
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :,
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.

        self.attn_score_limit = ScheduledFloat((0.0, 5.0), (5000.0, 20.0))
        self.attn_score_penalty_prob = ScheduledFloat((0.0, 1.0), (5000.0, 1.0), (5001.0, 0.1))

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim,
            bias=True, initial_scale=0.125 * query_head_dim**-0.25
        )

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )

        # the following are for diagnostics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 1, pos_dim)
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
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim, (
            p.shape[-1],
            num_heads,
            pos_head_dim,
        )

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(k) # does nothing in the forward pass.   [this may not really be needed due to the orthogonality constraint.]
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if True:
            # position scores.
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
                2, 0, 3, 1
            )
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            if torch.jit.is_tracing():
                (num_heads, batch_size, time1, n) = pos_scores.shape
                rows = torch.arange(start=time1 - 1, end=-1, step=-1)
                cols = torch.arange(seq_len)
                rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads, batch_size, time1, seq_len)
            else:
                pos_scores = pos_scores.as_strided(
                    (num_heads, batch_size, seq_len, seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (seq_len - 1),
                )

            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < float(self.attn_score_penalty_prob):
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(
                attn_scores, limit=float(self.attn_score_limit), penalty=1.0e-04, name=self.name
            )

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


class SelfAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

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
        self.in_proj = OrthogonalLinear(embed_dim, num_heads * value_head_dim,
                                        bias=True, out_groups=num_heads)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

        f = max(1.0, embed_dim / (num_heads * value_head_dim))
        # the whitening metric cannot be less than f because of the rank imposed
        # by the bottleneck.  the final whitening limit will be (2.0*3.0) times f,
        # i.e. 6 times greater than the mathematical smallest value it can have.
        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(f * 2.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
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
        x = self.whiten(x)

        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Zapformer model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
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
            activation="SwashL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.5,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05
        )

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """.
                Args:
                   x: a Tensor of shape (seq_len, batch_size, num_channels)
        attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
                Returns:
                   a Tensor with the same shape as x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.

        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x



class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zapformer model.
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
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.


        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(channels=bottleneck_dim, kernel_size=kernel_size)
            if causal
            else nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

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
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and chunk_size >= 0
        ):
            # Not support exporting a model for simulated streaming decoding
            assert (
                self.causal
            ), "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)

        return x


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
    time_embed_dim = 64

    c = Zapformer(
        input_dim=input_dim,
        encoder_dim=(64, 96),
        time_embed_dim=time_embed_dim,
        num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
    )

    batch_size = 6  # make it even, as PredictLoss requires even batch size.
    seq_len = 21
    # Just make sure the forward pass runs.
    time_embed = torch.randn(batch_size, time_embed_dim)

    f = c(
        torch.randn(seq_len, batch_size, input_dim),
        time_embed,
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f.sum().backward()
    c.eval()
    f = c(
        torch.randn(seq_len, batch_size, input_dim),
        time_embed,
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
