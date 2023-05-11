#!/usr/bin/env python3
<<<<<<< HEAD
# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
=======
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
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
<<<<<<< HEAD
import itertools
import logging
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
)
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    Identity,
    MaxEig,
    ScaledConv1d,
    Whiten,
    _diag,
    penalize_abs_values_gt,
    random_clamp,
    softmax,
)
from torch import Tensor, nn

from icefall.dist import get_rank
from icefall.utils import is_jit_tracing, make_pad_mask


class Zipformer(EncoderInterface):
    """
    Args:
        num_features (int): Number of input features
        d_model: (int,int): embedding dimension of 2 encoder stacks
        attention_dim: (int,int): attention dimension of 2 encoder stacks
        nhead (int, int): number of heads
        dim_feedforward (int, int): feedforward dimension in 2 encoder stacks
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        vgg_frontend (bool): whether to use vgg frontend.
        warmup_batches (float): number of batches to warm up over
    """

    def __init__(
        self,
        num_features: int,
        output_downsampling_factor: int = 2,
        encoder_dims: Tuple[int] = (384, 384),
        attention_dim: Tuple[int] = (256, 256),
        encoder_unmasked_dims: Tuple[int] = (256, 256),
        zipformer_downsampling_factors: Tuple[int] = (2, 4),
        nhead: Tuple[int] = (8, 8),
        feedforward_dim: Tuple[int] = (1536, 2048),
        num_encoder_layers: Tuple[int] = (12, 12),
        dropout: float = 0.1,
        cnn_module_kernels: Tuple[int] = (31, 31),
        pos_dim: int = 4,
        warmup_batches: float = 4000.0,
    ) -> None:
        super(Zipformer, self).__init__()

        self.num_features = num_features
        assert 0 < encoder_dims[0] <= encoder_dims[1]
        self.encoder_dims = encoder_dims
        self.encoder_unmasked_dims = encoder_unmasked_dims
        self.zipformer_downsampling_factors = zipformer_downsampling_factors
        self.output_downsampling_factor = output_downsampling_factor

        # will be written to, see set_batch_count()
        self.batch_count = 0
        self.warmup_end = warmup_batches

        for u, d in zip(encoder_unmasked_dims, encoder_dims):
            assert u <= d, (u, d)

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T - 7)//2, encoder_dims).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T - 7)//2
        #   (2) embedding: num_features -> encoder_dims
        self.encoder_embed = Conv2dSubsampling(
            num_features, encoder_dims[0], dropout=dropout
        )

        # each one will be ZipformerEncoder or DownsampledZipformerEncoder
        encoders = []

        num_encoders = len(encoder_dims)
        for i in range(num_encoders):
            encoder_layer = ZipformerEncoderLayer(
                encoder_dims[i],
                attention_dim[i],
                nhead[i],
                feedforward_dim[i],
                dropout,
                cnn_module_kernels[i],
                pos_dim,
=======
import math
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from encoder_interface import EncoderInterface
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ChunkCausalDepthwiseConv1d,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
)
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
        encoder_unmasked_dim (int or Tuple[int]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
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
        memory_dim: if supplied and >0, will be the dimension of the memory embeddings
            passed into the zipformer (e.g. this might be the output of another
            Zipformer used to create embedding vectors.)
    """
    def __init__(
            self,
            output_downsampling_factor: int = 2,
            downsampling_factor: Tuple[int] = (2, 4),
            encoder_dim: Union[int, Tuple[int]] = 384,
            num_encoder_layers: Union[int, Tuple[int]] = 4,
            encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
            query_head_dim: Union[int, Tuple[int]]  = 24,
            pos_head_dim: Union[int, Tuple[int]]  = 4,
            value_head_dim: Union[int, Tuple[int]] = 12,
            num_heads: Union[int, Tuple[int]] = 8,
            feedforward_dim: Union[int, Tuple[int]] = 1536,
            cnn_module_kernel: Union[int, Tuple[int]] = 31,
            memory_dim: int = -1,
            pos_dim: int = 192,
            dropout: FloatLike = None,  # see code below for default
            warmup_batches: float = 4000.0,
            causal: bool = False,
            chunk_size: Tuple[int] = [-1],
            left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3),
                                     (20000.0, 0.1))

        def _to_tuple(x):
            """ Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor # int
        self.downsampling_factor = downsampling_factor # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim) # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(encoder_unmasked_dim) # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        query_head_dim = _to_tuple(query_head_dim)
        value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u,d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):

            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                memory_dim=memory_dim,
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
<<<<<<< HEAD
            encoder = ZipformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
            )

            if zipformer_downsampling_factors[i] != 1:
                encoder = DownsampledZipformerEncoder(
                    encoder,
                    input_dim=encoder_dims[i - 1] if i > 0 else encoder_dims[0],
                    output_dim=encoder_dims[i],
                    downsample=zipformer_downsampling_factors[i],
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # initializes self.skip_layers and self.skip_modules
        self._init_skip_modules()

        self.downsample_output = AttentionDownsample(
            encoder_dims[-1], encoder_dims[-1], downsample=output_downsampling_factor
        )

    def _get_layer_skip_dropout_prob(self):
        if not self.training:
            return 0.0
        batch_count = self.batch_count
        min_dropout_prob = 0.025

        if batch_count > self.warmup_end:
            return min_dropout_prob
        else:
            return 0.5 - (batch_count / self.warmup_end) * (0.5 - min_dropout_prob)

    def _init_skip_modules(self):
        """
        If self.zipformer_downampling_factors = (1, 2, 4, 8, 4, 2), then at the input of layer
        indexed 4 (in zero indexing), with has subsapling_factor=4, we combine the output of
        layers 2 and 3; and at the input of layer indexed 5, which which has subsampling_factor=2,
        we combine the outputs of layers 1 and 5.
        """
        skip_layers = []
        skip_modules = []
        z = self.zipformer_downsampling_factors
        for i in range(len(z)):
            if i <= 1 or z[i - 1] <= z[i]:
                skip_layers.append(None)
                skip_modules.append(SimpleCombinerIdentity())
            else:
                # TEMP
                for j in range(i - 2, -1, -1):
                    if z[j] <= z[i] or j == 0:
                        # TEMP logging statement.
                        logging.info(
                            f"At encoder stack {i}, which has downsampling_factor={z[i]}, we will "
                            f"combine the outputs of layers {j} and {i-1}, with downsampling_factors={z[j]} and {z[i-1]}."
                        )
                        skip_layers.append(j)
                        skip_modules.append(
                            SimpleCombiner(
                                self.encoder_dims[j],
                                self.encoder_dims[i - 1],
                                min_weight=(0.0, 0.25),
                            )
                        )
                        break
        self.skip_layers = skip_layers
        self.skip_modules = nn.ModuleList(skip_modules)

    def get_feature_masks(self, x: torch.Tensor) -> List[float]:
        # Note: The actual return type is Union[List[float], List[Tensor]],
        # but to make torch.jit.script() work, we use List[float]
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all encoder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoder dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_downsampling_factors times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (num_frames, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dims)
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dims[0] == _encoder_dims0, (
            self.encoder_dims,
            _encoder_dims0,
        )

        max_downsampling_factor = max(self.zipformer_downsampling_factors)

        num_frames_max = num_frames0 + max_downsampling_factor - 1

        feature_mask_dropout_prob = 0.15

        # frame_mask_max shape: (num_frames_max, batch_size, 1)
        frame_mask_max = (
            torch.rand(num_frames_max, batch_size, 1, device=x.device)
            > feature_mask_dropout_prob
        ).to(x.dtype)

        feature_masks = []
        for i in range(num_encoders):
            ds = self.zipformer_downsampling_factors[i]
            upsample_factor = max_downsampling_factor // ds

            frame_mask = (
                frame_mask_max.unsqueeze(1)
                .expand(num_frames_max, upsample_factor, batch_size, 1)
                .reshape(num_frames_max * upsample_factor, batch_size, 1)
            )
            num_frames = (num_frames0 + ds - 1) // ds
            frame_mask = frame_mask[:num_frames]
            feature_mask = torch.ones(
                num_frames,
                batch_size,
                self.encoder_dims[i],
                dtype=x.dtype,
                device=x.device,
            )
            u = self.encoder_unmasked_dims[i]
            feature_mask[:, :, u:] *= frame_mask
=======
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(max(encoder_dim),
                                                  downsample=output_downsampling_factor,
                                                  dropout=dropout)

    def get_feature_masks(
            self,
            x: torch.Tensor) -> List[Union[float, Tensor]]:
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (1, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [ 1.0 ] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0

        feature_mask_dropout_prob = 0.125

        # mask1 shape: (1, batch_size, 1)
        mask1 = (torch.rand(1, batch_size, 1,
                            device=x.device) >
                 feature_mask_dropout_prob).to(x.dtype)

        # mask2 has additional sequences masked, about twice the number.
        mask2 = torch.logical_and(mask1,
                                  (torch.rand(1, batch_size, 1,
                                              device=x.device) >
                                   feature_mask_dropout_prob).to(x.dtype))


        # dim: (1, batch_size, 2)
        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.encoder_dim[i]
            feature_mask = torch.ones(1, batch_size, channels,
                                       dtype=x.dtype, device=x.device)
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
            feature_masks.append(feature_mask)

        return feature_masks

<<<<<<< HEAD
=======

    def get_chunk_info(self) -> Tuple[int, int]:
        """
        Returns chunk_size and left_context_chunks.
        """
        if not self.causal:
            return -1, -1
        chunk_size = random.choice(self.chunk_size)
        if chunk_size == -1:
            left_context_chunks = -1
        else:
            left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1
        return chunk_size, left_context_chunks


>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
<<<<<<< HEAD
=======
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
<<<<<<< HEAD
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, encoder_dims[-1])
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        lengths = (x_lens - 7) >> 1
        assert x.size(0) == lengths.max().item(), (x.shape, lengths, lengths.max())
        mask = make_pad_mask(lengths)

        outputs = []
        feature_masks = self.get_feature_masks(x)

        for i, (module, skip_module) in enumerate(
            zip(self.encoders, self.skip_modules)
        ):
            ds = self.zipformer_downsampling_factors[i]
            k = self.skip_layers[i]
            if isinstance(k, int):
                layer_skip_dropout_prob = self._get_layer_skip_dropout_prob()
                if torch.jit.is_scripting() or torch.jit.is_tracing():
                    x = skip_module(outputs[k], x)
                elif (not self.training) or random.random() > layer_skip_dropout_prob:
                    x = skip_module(outputs[k], x)
            x = module(
                x,
                feature_mask=feature_masks[i],
                src_key_padding_mask=None if mask is None else mask[..., ::ds],
            )
            outputs.append(x)

        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        lengths = (lengths + 1) >> 1

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths


class ZipformerEncoderLayer(nn.Module):
    """
    ZipformerEncoderLayer is made up of self-attn, feedforward and convolution networks.

    Args:
        d_model: the number of expected features in the input (required).
=======
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
             masked position. May be None.
          memory:  optionally, the memory embeddings of shape (memory_len, batch_size, memory_dim)
          memory_key_padding_mask: optionally the mask for padding of memory input (for source-
             attention), of shape  (batch_size, memory_len); True means
              masked position.  May be None.

        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        outputs = []
        feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        if self.training and memory is not None:
            batch_size = x.shape[1]
            # setting memory to zero should be equivalent to not using the
            # memory input at all, since the Attention module has no biases.
            memory_dropout_rate = 0.05
            memory = memory * (torch.rand(batch_size, 1, device=memory.device) >
                               memory_dropout_rate)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(x,
                       chunk_size=chunk_size,
                       feature_mask=feature_masks[i],
                       src_key_padding_mask=(None if src_key_padding_mask is None
                                             else src_key_padding_mask[...,::ds]),
                       attn_mask=attn_mask,
                       memory=memory,
                       memory_key_padding_mask=memory_key_padding_mask,
            )
            outputs.append(x)

        def get_full_dim_output():
            num_encoders = len(self.encoder_dim)
            assert len(outputs) == num_encoders
            output_dim = max(self.encoder_dim)
            output_pieces = [ outputs[-1] ]
            cur_dim = self.encoder_dim[-1]
            for i in range(num_encoders - 2, -1, -1):
                d = self.encoder_dim[i]
                if d > cur_dim:
                    this_output = outputs[i]
                    output_pieces.append(this_output[..., cur_dim:d])
                    cur_dim = d
            assert cur_dim == output_dim
            return torch.cat(output_pieces, dim=-1)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = get_full_dim_output()
        x = self.downsample_output(x)

        d = self.output_downsampling_factor
        lengths = (x_lens + d - 1) // d

        return x, lengths

    def _get_attn_mask(self, x: Tensor,
                       chunk_size: int,
                       left_context_chunks: int
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
            assert all (chunk_size * left_context_chunks >=
                        (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i]
                        for i in range(num_encoders))
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c,
                                     src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x),
                          (20000.0, ratio * x),
                          default=x)

def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))



class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
<<<<<<< HEAD
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
=======
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """
<<<<<<< HEAD

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        nhead: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        pos_dim: int = 4,
    ) -> None:
        super(ZipformerEncoderLayer, self).__init__()

        self.d_model = d_model

        # will be written to, see set_batch_count()
        self.batch_count = 0

        self.self_attn = RelPositionMultiheadAttention(
            d_model,
            attention_dim,
            nhead,
            pos_dim,
            dropout=0.0,
        )

        self.pooling = PoolingModule(d_model)

        self.feed_forward1 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.feed_forward2 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.conv_module1 = ConvolutionModule(d_model, cnn_module_kernel)

        self.conv_module2 = ConvolutionModule(d_model, cnn_module_kernel)

        self.norm_final = BasicNorm(d_model)

        self.bypass_scale = nn.Parameter(torch.tensor(0.5))

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.whiten = Whiten(
            num_groups=1, whitening_limit=5.0, prob=(0.025, 0.25), grad_scale=0.01
        )

    def get_bypass_scale(self):
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return self.bypass_scale
        if random.random() < 0.1:
            # ensure we get grads if self.bypass_scale becomes out of range
            return self.bypass_scale
        # hardcode warmup period for bypass scale
        warmup_period = 20000.0
        initial_clamp_min = 0.75
        final_clamp_min = 0.25
        if self.batch_count > warmup_period:
            clamp_min = final_clamp_min
        else:
            clamp_min = initial_clamp_min - (self.batch_count / warmup_period) * (
                initial_clamp_min - final_clamp_min
            )
        return self.bypass_scale.clamp(min=clamp_min, max=1.0)

    def get_dynamic_dropout_rate(self):
        # return dropout rate for the dynamic modules (self_attn, pooling, convolution); this
        # starts at 0.2 and rapidly decreases to 0.  Its purpose is to keep the training stable
        # at the beginning, by making the network focus on the feedforward modules.
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return 0.0
        warmup_period = 2000.0
        initial_dropout_rate = 0.2
        final_dropout_rate = 0.0
        if self.batch_count > warmup_period:
            return final_dropout_rate
        else:
            return initial_dropout_rate - (
                initial_dropout_rate * final_dropout_rate
            ) * (self.batch_count / warmup_period)
=======
    def __init__(
            self,
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            value_head_dim: int,
            feedforward_dim: int,
            dropout: FloatLike = 0.1,
            cnn_module_kernel: int = 31,
            causal: bool = False,
            memory_dim: int = -1,
            attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
            ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate,
                                   straight_through_rate=0.025)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0.025)


        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim=pos_dim, num_heads=num_heads,
            query_head_dim=query_head_dim, pos_head_dim=pos_head_dim,
            dropout=0.0,
        )


        self.self_attn1 = Attention(embed_dim, embed_dim, num_heads,
                                        value_head_dim)

        self.self_attn2 = Attention(embed_dim, embed_dim, num_heads,
                                    value_head_dim)

        if memory_dim > 0:
            self.attn_weights = MultiheadAttentionWeights(
                memory_dim, embed_dim,
                num_heads=num_heads,
                head_dim=query_head_dim,
                dropout=0.0,
            )
            self.src_attn1 = Attention(memory_dim, embed_dim, num_heads,
                                       value_head_dim)
            self.src_attn2 = Attention(memory_dim, embed_dim, num_heads,
                                       value_head_dim)


        self.feed_forward1 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 3) // 4,
                                               dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim,
                                               feedforward_dim,
                                               dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 5) // 4,
                                               dropout)

        self.nonlin_attention = NonlinAttention(embed_dim,
                                                hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim,
                                             cnn_module_kernel,
                                             causal=causal)

        self.conv_module2 = ConvolutionModule(embed_dim,
                                              cnn_module_kernel,
                                              causal=causal)


        #self.attention_squeeze = AttentionSqueeze(embed_dim, embed_dim // 2)

        self.norm = BiasNorm(embed_dim)

        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.balancer1 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.2, max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(4.0, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.balancer2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.1, max_abs=4.0,
        )


    def get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.bypass_min),
                                    max=float(self.bypass_max))
            layer_skip_rate = float(self.layer_skip_rate)
            if layer_skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > layer_skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            return ans

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask


    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
<<<<<<< HEAD
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        batch_split: if not None, this layer will only be applied to

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        src_orig = src

        # macaron style feed forward module
        src = src + self.feed_forward1(src)

        # dropout rate for submodules that interact with time.
        dynamic_dropout = self.get_dynamic_dropout_rate()

        # pooling module
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            src = src + self.pooling(src, key_padding_mask=src_key_padding_mask)
        elif random.random() >= dynamic_dropout:
            src = src + self.pooling(src, key_padding_mask=src_key_padding_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            src_att, attn_weights = self.self_attn(
                src,
                pos_emb=pos_emb,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + src_att

            src = src + self.conv_module1(
                src, src_key_padding_mask=src_key_padding_mask
            )

            src = src + self.feed_forward2(src)

            src = src + self.self_attn.forward2(src, attn_weights)

            src = src + self.conv_module2(
                src, src_key_padding_mask=src_key_padding_mask
            )
        else:
            use_self_attn = random.random() >= dynamic_dropout
            if use_self_attn:
                src_att, attn_weights = self.self_attn(
                    src,
                    pos_emb=pos_emb,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                )
                src = src + src_att

            if random.random() >= dynamic_dropout:
                src = src + self.conv_module1(
                    src, src_key_padding_mask=src_key_padding_mask
                )

            src = src + self.feed_forward2(src)
            if use_self_attn:
                src = src + self.self_attn.forward2(src, attn_weights)

            if random.random() >= dynamic_dropout:
                src = src + self.conv_module2(
                    src, src_key_padding_mask=src_key_padding_mask
                )

        src = src + self.feed_forward3(src)

        src = self.norm_final(self.balancer(src))

        delta = src - src_orig

        src = src_orig + delta * self.get_bypass_scale()

        return self.whiten(src)


class ZipformerEncoder(nn.Module):
    r"""ZipformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ZipformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
        >>> zipformer_encoder = ZipformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
    ) -> None:
        super().__init__()
        # will be written to, see set_batch_count() Note: in inference time this
        # may be zero but should be treated as large, we can check if
        # self.training is true.
        self.batch_count = 0
        self.warmup_begin = warmup_begin
        self.warmup_end = warmup_end
        # module_seed is for when we need a random number that is unique to the module but
        # shared across jobs.   It's used to randomly select how many layers to drop,
        # so that we can keep this consistent across worker tasks (for efficiency).
        self.module_seed = torch.randint(0, 1000, ()).item()

        self.encoder_pos = RelPositionalEncoding(encoder_layer.d_model, dropout)
=======
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
         pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
         chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
       feature_mask: something that broadcasts with src, that we'll multiply `src`
              by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
         attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
               True means masked position. May be None.
    src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
             masked position.  May be None.

        Returns:
           A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        if memory is not None and hasattr(self, 'attn_weights'):
            src_attn_weights = self.attn_weights(memory, src, memory_key_padding_mask)

        src = src + self.feed_forward1(src)

        attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        if True:
            selected_attn_weights = attn_weights[0:2]
            if random.random() < float(self.const_attention_rate):
                # Make attention weights constant.  The intention is to
                # encourage these modules to do something similar to an
                # averaging-over-time operation.
                # only need the mask, can just use the 1st one and expand later
                selected_attn_weights = selected_attn_weights[0:1]
                selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
                selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))
                selected_attn_weights = selected_attn_weights.expand(2, -1, -1, -1)


        na = self.balancer_na(self.nonlin_attention(src,
                                                    selected_attn_weights[0:1]))

        src = src + (na if attn_dropout_mask is None else na * attn_dropout_mask)

        self_attn = self.self_attn1(
            src, attn_weights)

        src = src + (self_attn if attn_dropout_mask is None else self_attn * attn_dropout_mask)

        if memory is not None and hasattr(self, 'attn_weights'):
            src = src + self.sequence_dropout(self.src_attn1(memory, src_attn_weights),
                                              attention_skip_rate)

        src = src + self.sequence_dropout(self.conv_module1(src, chunk_size=chunk_size,
                                                            src_key_padding_mask=src_key_padding_mask),
                                          float(self.conv_skip_rate))

        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)),
                                          float(self.ff2_skip_rate))

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(
            src, attn_weights)

        src = src + (self_attn if attn_dropout_mask is None else self_attn * attn_dropout_mask)

        if memory is not None and hasattr(self, 'attn_weights'):
            src = src + self.sequence_dropout(self.src_attn2(memory, src_attn_weights),
                                              attention_skip_rate)

        src = src + self.sequence_dropout(self.conv_module2(src, chunk_size=chunk_size,
                                                            src_key_padding_mask=src_key_padding_mask),
                                          float(self.conv_skip_rate))

        src = src + self.sequence_dropout(self.balancer_ff3(self.feed_forward3(src)),
                                          float(self.ff3_skip_rate))

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src

class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
     encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
           pos_dim: the dimension for the relative positional encoding

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
            pos_dim: int,
            dropout: float,
            warmup_begin: float,
            warmup_end: float,
            initial_layerdrop_rate: float = 0.5,
            final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15,
                                                        length_factor=1.0)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

<<<<<<< HEAD
        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_layers):
            self.layers[i].warmup_begin = cur_begin
            cur_begin += delta
            self.layers[i].warmup_end = cur_begin

    def get_layers_to_drop(self, rnd_seed: int):
        ans = set()
        if not self.training:
            return ans

        batch_count = self.batch_count
        num_layers = len(self.layers)

        def get_layerdrop_prob(layer: int) -> float:
            layer_warmup_begin = self.layers[layer].warmup_begin
            layer_warmup_end = self.layers[layer].warmup_end

            initial_layerdrop_prob = 0.5
            final_layerdrop_prob = 0.05

            if batch_count == 0:
                # As a special case, if batch_count == 0, return 0 (drop no
                # layers).  This is rather ugly, I'm afraid; it is intended to
                # enable our scan_pessimistic_batches_for_oom() code to work correctly
                # so if we are going to get OOM it will happen early.
                # also search for 'batch_count' with quotes in this file to see
                # how we initialize the warmup count to a random number between
                # 0 and 10.
                return 0.0
            elif batch_count < layer_warmup_begin:
                return initial_layerdrop_prob
            elif batch_count > layer_warmup_end:
                return final_layerdrop_prob
            else:
                # linearly interpolate
                t = (batch_count - layer_warmup_begin) / layer_warmup_end
                assert 0.0 <= t < 1.001, t
                return initial_layerdrop_prob + t * (
                    final_layerdrop_prob - initial_layerdrop_prob
                )

        shared_rng = random.Random(batch_count + self.module_seed)
        independent_rng = random.Random(rnd_seed)

        layerdrop_probs = [get_layerdrop_prob(i) for i in range(num_layers)]
        tot = sum(layerdrop_probs)
        # Instead of drawing the samples independently, we first randomly decide
        # how many layers to drop out, using the same random number generator between
        # jobs so that all jobs drop out the same number (this is for speed).
        # Then we use an approximate approach to drop out the individual layers
        # with their specified probs while reaching this exact target.
        num_to_drop = int(tot) + int(shared_rng.random() < (tot - int(tot)))

        layers = list(range(num_layers))
        independent_rng.shuffle(layers)

        # go through the shuffled layers until we get the required number of samples.
        if num_to_drop > 0:
            for layer in itertools.cycle(layers):
                if independent_rng.random() < layerdrop_probs[layer]:
                    ans.add(layer)
                if len(ans) == num_to_drop:
                    break
        if shared_rng.random() < 0.005 or __name__ == "__main__":
            logging.info(
                f"warmup_begin={self.warmup_begin:.1f}, warmup_end={self.warmup_end:.1f}, "
                f"batch_count={batch_count:.1f}, num_to_drop={num_to_drop}, layers_to_drop={ans}"
            )
        return ans
=======
        assert 0 <= warmup_begin <= warmup_end

        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat((cur_begin, initial_layerdrop_rate),
                                                             (cur_end, final_layerdrop_rate),
                                                             default=0.0)
            cur_begin = cur_end
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

    def forward(
        self,
        src: Tensor,
<<<<<<< HEAD
        # Note: The type of feature_mask should be Union[float, Tensor],
        # but to make torch.jit.script() work, we use `float` here
        feature_mask: float = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
=======
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
<<<<<<< HEAD
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: (x, x_no_combine), both of shape (S, N, E)
=======
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.
            memory:  optionally, the memory embeddings of shape (memory_len, batch_size, memory_dim)
            memory_key_padding_mask: optionally the mask for padding of memory input (for source-
                attention), of shape  (batch_size, memory_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        """
        pos_emb = self.encoder_pos(src)
        output = src

<<<<<<< HEAD
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            layers_to_drop = []
        else:
            rnd_seed = src.numel() + random.randint(0, 1000)
            layers_to_drop = self.get_layers_to_drop(rnd_seed)
=======
        rnd_seed = src.numel() + random.randint(0, 1000)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
<<<<<<< HEAD
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                if i in layers_to_drop:
                    continue
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
=======
            output = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
            )

            output = output * feature_mask

        return output


<<<<<<< HEAD
class DownsampledZipformerEncoder(nn.Module):
    r"""
    DownsampledZipformerEncoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """

    def __init__(
        self, encoder: nn.Module, input_dim: int, output_dim: int, downsample: int
    ):
        super(DownsampledZipformerEncoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = AttentionDownsample(input_dim, output_dim, downsample)
        self.encoder = encoder
        self.upsample = SimpleUpsample(output_dim, downsample)
        self.out_combiner = SimpleCombiner(
            input_dim, output_dim, min_weight=(0.0, 0.25)
        )

    def forward(
        self,
        src: Tensor,
        # Note: the type of feature_mask should be Unino[float, Tensor],
        # but to make torch.jit.script() happ, we use float here
        feature_mask: float = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            mask: the mask for the src sequence (optional).  CAUTION: we need to downsample
                  this, if we are to support it.  Won't work correctly yet.
            src_key_padding_mask: the mask for the src keys per batch (optional).  Should
                  be downsampled already.

        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
=======
class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """
    def __init__(
            self,
            embed_dim: int,
            skip_rate: FloatLike = 0.0,
            straight_through_rate: FloatLike = 0.0,
            scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
            scale_max: FloatLike = 1.0):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)


    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.scale_min),
                                    max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))

            return ans

    def forward(self,
                src_orig: Tensor,
                src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig)  * bypass_scale




class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """
    def __init__(self,
                 encoder: nn.Module,
                 dim: int,
                 downsample: int,
                 dropout: FloatLike):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim,
                                           downsample, dropout)
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0.025)


    def forward(self,
                src: Tensor,
                chunk_size: int = -1,
                feature_mask: Union[Tensor, float] = 1.0,
                attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.
            memory:  optionally, the memory embeddings of shape (memory_len, batch_size, memory_dim)
            memory_key_padding_mask: optionally the mask for padding of memory input (for source-
                attention), of shape  (batch_size, memory_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
<<<<<<< HEAD
        if mask is not None:
            mask = mask[::ds, ::ds]

        src = self.encoder(
            src,
            feature_mask=feature_mask,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]
=======
        if attn_mask is not None:
            attn_mask = attn_mask[::ds,::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[:src_orig.shape[0]]
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        return self.out_combiner(src_orig, src)


<<<<<<< HEAD
class AttentionDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: int):
        """
        Require out_channels > in_channels.
        """
        super(AttentionDownsample, self).__init__()
        self.query = nn.Parameter(torch.randn(in_channels) * (in_channels**-0.5))

        # fill in the extra dimensions with a projection of the input
        if out_channels > in_channels:
            self.extra_proj = nn.Linear(
                in_channels * downsample, out_channels - in_channels, bias=False
            )
        else:
            self.extra_proj = None
        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
=======

class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """
    def __init__(self,
                 channels: int,
                 downsample: int,
                 dropout: FloatLike):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

<<<<<<< HEAD
        # Pad to an exact multiple of self.downsample, could be 0 for onnx-export-compatibility
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds, (src.shape[0], d_seq_len, ds)

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        scores = (src * self.query).sum(dim=-1, keepdim=True)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            scores = penalize_abs_values_gt(scores, limit=10.0, penalty=1.0e-04)

        weights = scores.softmax(dim=1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)
        src = src.permute(0, 2, 1, 3).reshape(d_seq_len, batch_size, ds * in_channels)

        if self.extra_proj is not None:
            ans2 = self.extra_proj(src)
            ans = torch.cat((ans, ans2), dim=2)
=======
        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
            # right-pad src, repeating the last element.
            pad = d_seq_len * ds - seq_len
            src_extra = src[src.shape[0]-1:].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
            assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """
<<<<<<< HEAD

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.bias = nn.Parameter(torch.randn(upsample, num_channels) * 0.01)

    def forward(self, src: Tensor) -> Tensor:
=======
    def __init__(self,
                 num_channels: int,
                 upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self,
                src: Tensor) -> Tensor:
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
<<<<<<< HEAD
        upsample = self.bias.shape[0]
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src + self.bias.unsqueeze(1)
=======
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


<<<<<<< HEAD
class SimpleCombinerIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        return src1


class SimpleCombiner(torch.nn.Module):
    """
    A very simple way of combining 2 vectors of 2 different dims, via a
    learned weighted combination in the shared part of the dim.
    Args:
         dim1: the dimension of the first input, e.g. 256
         dim2: the dimension of the second input, e.g. 384.
    The output will have the same dimension as dim2.
    """

    def __init__(self, dim1: int, dim2: int, min_weight: Tuple[float] = (0.0, 0.0)):
        super(SimpleCombiner, self).__init__()
        assert dim2 >= dim1, (dim2, dim1)
        self.weight1 = nn.Parameter(torch.zeros(()))
        self.min_weight = min_weight

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        """
        src1: (*, dim1)
        src2: (*, dim2)

        Returns: a tensor of shape (*, dim2)
        """
        assert src1.shape[:-1] == src2.shape[:-1], (src1.shape, src2.shape)

        weight1 = self.weight1
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if (
                self.training
                and random.random() < 0.25
                and self.min_weight != (0.0, 0.0)
            ):
                weight1 = weight1.clamp(
                    min=self.min_weight[0], max=1.0 - self.min_weight[1]
                )

        src1 = src1 * weight1
        src2 = src2 * (1.0 - weight1)

        src1_dim = src1.shape[-1]
        src2_dim = src2.shape[-1]
        if src1_dim != src2_dim:
            if src1_dim < src2_dim:
                src1 = torch.nn.functional.pad(src1, (0, src2_dim - src1_dim))
            else:
                src1 = src1[:src2_dim]

        return src1 + src2


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct a PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        if is_jit_tracing():
            # 10k frames correspond to ~100k ms, e.g., 100 seconds, i.e.,
            # It assumes that the maximum input won't have more than
            # 10k frames.
            #
            # TODO(fangjun): Use torch.jit.script() for this module
            max_len = 10000
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(max_len))

=======
class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works done by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """
    def __init__(
        self, embed_dim: int,
            dropout_rate: FloatLike,
            max_len: int = 1000,
            length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))



>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
<<<<<<< HEAD
            if self.pe.size(1) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(0), self.d_model)
        pe_negative = torch.zeros(x.size(0), self.d_model)
        position = torch.arange(0, x.size(0), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tensor:
        """Add positional encoding.
=======
            if self.pe.size(0) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        T = x.size(0)
        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T-1), T,
                         device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = (self.embed_dim ** 0.5)
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = compression_length * x.sign() * ((x.abs() + compression_length).log() - math.log(compression_length))

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan() # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)


    def forward(self, x: torch.Tensor) -> Tensor:
        """Create positional encoding.
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        Args:
            x (torch.Tensor): Input tensor (time, batch, `*`).

        Returns:
<<<<<<< HEAD
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).
=======
            positional embedding, of shape (1, 2*time-1, `*`).
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        """
        self.extend_pe(x)
        pos_emb = self.pe[
<<<<<<< HEAD
            :,
            self.pe.size(1) // 2
            - x.size(0)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(0),
        ]
        return self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding
=======
            self.pe.size(0) // 2
            - x.size(0)
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)



class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
<<<<<<< HEAD
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module, may be less or more than embed_dim
            but must be a multiple of num_heads.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        pos_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = attention_dim // num_heads
        self.pos_dim = pos_dim
        assert self.head_dim % 2 == 0, self.head_dim
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim,
            num_heads,
            attention_dim,
        )

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5, dividing it between the query and key.
        in_proj_dim = (
            2 * attention_dim  # query, key
            + attention_dim // 2  # value
            + pos_dim * num_heads  # positional encoding query
        )

        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=self.head_dim**-0.25
        )

        # self.whiten_values is applied on the values in forward();
        # it just copies the keys but prevents low-rank distribution by modifying grads.
        self.whiten_values = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )
        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            embed_dim, num_heads * pos_dim, bias=False, initial_scale=0.05
        )

        # the following are for diagnosics only, see --print-diagnostics option.
        # they only copy their inputs.
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

        self.out_proj = ScaledLinear(
            attention_dim // 2, embed_dim, bias=True, initial_scale=0.05
        )

        self.in_proj2 = nn.Linear(embed_dim, attention_dim // 2, bias=False)
        self.out_proj2 = ScaledLinear(
            attention_dim // 2, embed_dim, bias=True, initial_scale=0.05
        )
        # self.whiten_values2 is applied on the values in forward2()
        self.whiten_values2 = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )
=======
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
            pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                          (4000.0, 0.0))
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True,
                                    initial_scale=query_head_dim**-0.25)

        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=_whitening_schedule(3.0),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(key_head_dim * num_heads,
                                     channel_dim=-1,
                                     min_positive=0.4,
                                     max_positive=0.6,
                                     min_abs=0.0,
                                     max_abs=100.0,
                                     prob=0.025)


        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(pos_dim,
                                       num_heads * pos_head_dim,
                                       bias=False,
                                       initial_scale=0.05)


        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
<<<<<<< HEAD
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x: input to be projected to query, key, value
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - x: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Returns: (attn_output, attn_weights)

             - attn_output: :math:`(S, N, E)` where S is the sequence length, N is the batch size,
                E is the embedding dimension.
              - attn_weights: :math:`(N * N, S, S)` where N is the batch size, H is the num-heads
                 and S is the sequence length.
        """
        x, weights = self.multi_head_attention_forward(
            self.in_proj(x),
            self.linear_pos(pos_emb),
            self.attention_dim,
            self.num_heads,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        return x, weights

    def multi_head_attention_forward(
        self,
        x_proj: Tensor,
        pos: Tensor,
        attention_dim: int,
        num_heads: int,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x_proj: the projected input, to be split into query, key, value.
            pos: head-specific biases arising from the positional embeddings.
            attention_dim: dimension inside attention mechanism
            num_heads: parallel attention heads.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - x: :math:`(L, N, 7 * A // 2)` where L is the target sequence length, N is the batch size, A is
              the attention dimension.  Will be split into (query, key, value, pos).
            - pos: :math:`(N, 2*L-1, A//2)` or :math:`(1, 2*L-1, A//2)` where L is the sequence
              length, N is the batch size, and A is the attention dim.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_weights: :math:`(N * H, S, S)` where N is the batch size,
              H is the num-heads, S is the sequence length.
        """

        seq_len, bsz, _ = x_proj.size()

        head_dim = attention_dim // num_heads
        pos_dim = self.pos_dim  # positional-encoding dim per head
        assert (
            head_dim * num_heads == attention_dim
        ), f"attention_dim must be divisible by num_heads: {head_dim}, {num_heads}, {attention_dim}"

        # self-attention
        q = x_proj[..., 0:attention_dim]
        k = x_proj[..., attention_dim : 2 * attention_dim]
        value_dim = attention_dim // 2
        v = x_proj[..., 2 * attention_dim : 2 * attention_dim + value_dim]
        # p is the position-encoding query, its dimension is num_heads*pos_dim..
        p = x_proj[..., 2 * attention_dim + value_dim :]

        k = self.whiten_keys(k)  # does nothing in the forward pass.
        v = self.whiten_values(v)  # does nothing in the forward pass.
        q = self.copy_query(q)  # for diagnostics only, does nothing.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, seq_len, seq_len]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    seq_len,
                    seq_len,
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.reshape(seq_len, bsz, num_heads, head_dim)
        p = p.reshape(seq_len, bsz, num_heads, pos_dim)
        k = k.reshape(seq_len, bsz, num_heads, head_dim)
        v = v.reshape(seq_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == seq_len, "{} == {}".format(
                key_padding_mask.size(1), seq_len
            )

        q = q.permute(1, 2, 0, 3)  # (batch, head, time1, head_dim)
        p = p.permute(1, 2, 0, 3)  # (batch, head, time1, pos_dim)
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)

        seq_len2 = 2 * seq_len - 1
        pos = pos.reshape(1, seq_len2, num_heads, pos_dim).permute(0, 2, 3, 1)
        # pos shape now: (batch, head, pos_dim, seq_len2)

        # (batch, head, time1, pos_dim) x (1, head, pos_dim, seq_len2) -> (batch, head, time1, seq_len2)
        #  [where seq_len2 represents relative position.]
        pos_weights = torch.matmul(p, pos)
        # the following .as_strided() expression converts the last axis of pos_weights from relative
        # to absolute position.  I don't know whether I might have got the time-offsets backwards or
        # not, but let this code define which way round it is supposed to be.
        if torch.jit.is_tracing():
            (batch_size, num_heads, time1, n) = pos_weights.shape
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(seq_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_weights = pos_weights.reshape(-1, n)
            pos_weights = torch.gather(pos_weights, dim=1, index=indexes)
            pos_weights = pos_weights.reshape(batch_size, num_heads, time1, seq_len)
        else:
            pos_weights = pos_weights.as_strided(
                (bsz, num_heads, seq_len, seq_len),
                (
                    pos_weights.stride(0),
                    pos_weights.stride(1),
                    pos_weights.stride(2) - pos_weights.stride(3),
                    pos_weights.stride(3),
                ),
                storage_offset=pos_weights.stride(3) * (seq_len - 1),
            )

        # caution: they are really scores at this point.
        attn_output_weights = torch.matmul(q, k) + pos_weights

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if training and random.random() < 0.1:
                # This is a harder way of limiting the attention scores to not be too large.
                # It incurs a penalty if any of them has an absolute value greater than 50.0.
                # this should be outside the normal range of the attention scores.  We use
                # this mechanism instead of, say, a limit on entropy, because once the entropy
                # gets very small gradients through the softmax can become very small, and
                # some mechanisms like that become ineffective.
                attn_output_weights = penalize_abs_values_gt(
                    attn_output_weights, limit=25.0, penalty=1.0e-04
                )

        # attn_output_weights: (batch, head, time1, time2)
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, seq_len, seq_len
        )

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights = attn_output_weights.masked_fill(
                    attn_mask, float("-inf")
                )
            else:
                attn_output_weights = attn_output_weights + attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, seq_len, seq_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, seq_len, seq_len
            )

        # Using this version of softmax, defined in scaling.py,
        # should save a little of the memory used in backprop by, if
        # we are in automatic mixed precision mode (amp) == autocast,
        # only storing the half-precision output for backprop purposes.
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        # If we are using chunk-wise attention mask and setting a limited
        # num_left_chunks, the attention may only see the padding values which
        # will also be masked out by `key_padding_mask`. At this circumstances,
        # the whole column of `attn_output_weights` will be `-inf`
        # (i.e. be `nan` after softmax). So we fill `0.0` at the masking
        # positions to avoid invalid loss value below.
        if (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and key_padding_mask is not None
        ):
            if attn_mask.size(0) != 1:
                attn_mask = attn_mask.view(bsz, num_heads, seq_len, seq_len)
                combined_mask = attn_mask | key_padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                # attn_mask.shape == (1, tgt_len, src_len)
                combined_mask = attn_mask.unsqueeze(0) | key_padding_mask.unsqueeze(
                    1
                ).unsqueeze(2)

            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, seq_len, seq_len
            )
            attn_output_weights = attn_output_weights.masked_fill(combined_mask, 0.0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, seq_len, seq_len
            )

        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, seq_len, head_dim // 2]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, attention_dim // 2)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        return attn_output, attn_output_weights

    def forward2(
=======
        chunk_size: int = -1,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 2, pos_dim)
           chunk_size
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

        q = x[...,0:query_dim]
        k = x[...,query_dim:2*query_dim]
        # p is the position-encoding query
        p = x[...,2*query_dim:]
        assert p.shape[-1] == num_heads * pos_head_dim


        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.


        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if not self.training or random.random() >= float(self.pos_emb_skip_rate):
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            pos_scores = pos_scores.as_strided((num_heads, batch_size, seq_len, seq_len),
                                               (pos_scores.stride(0),
                                                pos_scores.stride(1),
                                                pos_scores.stride(2)-pos_scores.stride(3),
                                                pos_scores.stride(3)),
                                               storage_offset=pos_scores.stride(3) * (seq_len - 1))

            attn_scores = attn_scores + pos_scores

        if self.training and random.random() < 0.1:
            # This is away of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 25.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores,
                                                 limit=25.0,
                                                 penalty=1.0e-04,
                                                 name=self.name)

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, seq_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights


    def _print_attn_entropy(
            self,
            attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1,2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


class Attention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim_in: the input embedding dimension
          embed_dim_out: the output embedding dimension (normally the same as input)
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """
    def __init__(
            self,
            embed_dim_in: int,
            embed_dim_out: int,
            num_heads: int,
            value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim_in,
                                 num_heads * value_head_dim,
                                 bias=False)

        self.out_proj = ScaledLinear(num_heads * value_head_dim,
                                     embed_dim_out, bias=False,
                                     initial_scale=0.05)

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)


    def forward(
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
<<<<<<< HEAD
        Second forward function, where we re-use the attn_weights returned by the first forward function
        but with different input.
        Args:
               x: input, of shape (seq_len, batch_size, embed_dim)
           attn_weights: attention weights returned by forward(), of shape (batch_size * num_heads, seq_len, seq_len)
        Returns:
               output of the same shape as x, i.e. (seq_len, batch_size, embed_dim)
        """
        num_heads = self.num_heads
        (seq_len, bsz, embed_dim) = x.shape
        head_dim = self.attention_dim // num_heads
        # v: (tgt_len, bsz, embed_dim // 2)
        v = self.in_proj2(x)
        v = self.whiten_values2(v)  # does nothing in the forward pass.
        v = v.reshape(seq_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        # now v: (bsz * num_heads, seq_len, head_dim // 2)
        attn_output = torch.bmm(attn_weights, v)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if random.random() < 0.001 or __name__ == "__main__":
                self._print_attn_stats(attn_weights, attn_output)

        # attn_output: (bsz * num_heads, seq_len, head_dim)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, self.attention_dim // 2)
        )
        # returned value is of shape (seq_len, bsz, embed_dim), like x.
        return self.out_proj2(attn_output)

    def _print_attn_stats(self, attn_weights: Tensor, attn_output: Tensor):
        # attn_weights: (batch_size * num_heads, seq_len, seq_len)
        # attn_output: (bsz * num_heads, seq_len, head_dim)
        (n, seq_len, head_dim) = attn_output.shape
        num_heads = self.num_heads
        bsz = n // num_heads
=======
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, query_len, key_len),
          Expect attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (num_heads, batch_size, query_len, key_len) = attn_weights.shape

        x = self.in_proj(x)     #  (key_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(key_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, key_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, query_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(
            query_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (query_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x


class MultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head cross-attention weights.  Allows src and target
    to have different dims.

    Args:
          key_embed_dim: number of channels of the thing that we'll project to
              make the query (corresponds to source).  e.g. 256
          query_embed_dim: number of channels of the thing that we'll project to
              make the query (corresponds to target).  e.g. 256
          num_heads:  number of heads to compute weights for, e.g. 8
           head_dim: dimension of the query and key, per head.  e.g. 24.
             dropout: dropout probability for attn_output_weights. Default: 0.0.
    """

    def __init__(
            self,
            key_embed_dim: int,
            query_embed_dim: int,
            num_heads: int,
            head_dim: int,
            dropout: float = 0.0,

    ) -> None:
        super().__init__()
        self.key_embed_dim = key_embed_dim
        self.query_embed_dim = query_embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.


        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.query_in_proj = ScaledLinear(query_embed_dim,
                                          head_dim * num_heads,
                                          bias=True,
                                          initial_scale=head_dim ** -0.25)

        # weights produced by this module are invariant to adding a constant to
        # the keys, so we don't need a bias for the keys.
        self.key_in_proj = ScaledLinear(key_embed_dim,
                                        head_dim * num_heads,
                                        bias=False,
                                        initial_scale=head_dim ** -0.25)

        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=_whitening_schedule(3.0),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)



    def forward(
        self,
        key: Tensor,
        query: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
              key: input of shape (key_len, batch_size, key_embed_dim)
            query: input of shape (query_len, batch_size, query_embed_dim)
          key_padding_mask: an optional bool tensor of shape (batch_size, key_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, query_len, key_len)
        """
        q = self.query_in_proj(query)
        k = self.key_in_proj(key)

        head_dim = self.head_dim
        num_heads = self.num_heads

        query_len, batch_size, _ = q.shape
        key_len, _batch_size, _ = k.shape
        assert _batch_size == batch_size

        k = self.whiten_keys(k)   # does nothing in the forward pass.

        q = q.reshape(query_len, batch_size, num_heads, head_dim)
        k = k.reshape(key_len, batch_size, num_heads, head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if self.training and random.random() < 0.1:
            # This is a way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 25.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores,
                                                 limit=25.0,
                                                 penalty=1.0e-04,
                                                 name=self.name)

        assert attn_scores.shape == (num_heads, batch_size, query_len, key_len)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, key_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights


    def _print_attn_entropy(
            self,
            attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
<<<<<<< HEAD
                attn_output = attn_output.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .reshape(bsz, num_heads, seq_len)
                    .mean(dim=(0, 2))
                )
                attn_output = attn_output.reshape(bsz, num_heads, seq_len, head_dim)
                attn_output = attn_output.permute(1, 0, 2, 3).reshape(
                    num_heads, bsz * seq_len, head_dim
                )
                attn_output_mean = attn_output.mean(dim=1, keepdim=True)
                attn_output = attn_output - attn_output_mean
                attn_covar = torch.matmul(attn_output.transpose(1, 2), attn_output) / (
                    bsz * seq_len
                )
                # attn_covar: (num_heads, head_dim, head_dim)
                # eigs, _ = torch.symeig(attn_covar)
                # logging.info(f"attn_weights_entropy = {attn_weights_entropy}, output_eigs = {eigs}")

                attn_covar = _diag(attn_covar).mean(dim=1)  # (num_heads,)
                embed_dim = self.in_proj2.weight.shape[1]
                in_proj_covar = (
                    self.in_proj2.weight.reshape(num_heads, head_dim, embed_dim) ** 2
                ).mean(dim=(1, 2))
                out_proj_covar = (
                    self.out_proj2.weight.reshape(embed_dim, num_heads, head_dim) ** 2
                ).mean(dim=(0, 2))
                logging.info(
                    f"attn_weights_entropy = {attn_weights_entropy}, covar={attn_covar}, in_proj_covar={in_proj_covar}, out_proj_covar={out_proj_covar}"
                )


class PoolingModule(nn.Module):
    """
    Averages the input over the time dimension and project with a square matrix.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = ScaledLinear(d_model, d_model, initial_scale=0.1, bias=False)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        """
        Args:
           x: a Tensor of shape (T, N, C)
          key_padding_mask: a Tensor of bool, of shape (N, T), with True in masked
            positions.
        Returns:
           a Tensor of shape (1, N, C)
        """
        if key_padding_mask is not None:
            if torch.jit.is_tracing():
                pooling_mask = (~key_padding_mask).to(x.dtype)
            else:
                pooling_mask = key_padding_mask.logical_not().to(x.dtype)  # (N, T)
            pooling_mask = pooling_mask / pooling_mask.sum(dim=1, keepdim=True)
            pooling_mask = pooling_mask.transpose(0, 1).contiguous().unsqueeze(-1)
            # now pooling_mask: (T, N, 1)
            x = (x * pooling_mask).sum(dim=0, keepdim=True)
        else:
            num_frames = x.shape[0]
            pooling_mask = 1.0 / num_frames
            x = (x * pooling_mask).sum(dim=0, keepdim=True)

        x = self.proj(x)
        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer model."""

    def __init__(self, d_model: int, feedforward_dim: int, dropout: float):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(d_model, feedforward_dim)
        self.balancer = ActivationBalancer(
            feedforward_dim, channel_dim=-1, max_abs=10.0, min_prob=0.25
        )
        self.activation = DoubleSwish()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = ScaledLinear(feedforward_dim, d_model, initial_scale=0.01)

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.balancer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
=======
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1,2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")



class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model.
    """
    def __init__(self,
                 embed_dim: int,
                 feedforward_dim: int,
                 dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(feedforward_dim,
                                        channel_dim=-1,
                                        min_positive=0.3,
                                        max_positive=1.0,
                                        min_abs=0.75,
                                        max_abs=5.0)

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(feedforward_dim, embed_dim,
                                                   activation='SwooshL',
                                                   dropout_p=dropout,
                                                   dropout_shared_dim=0, bias=True,
                                                   initial_scale=0.1)

        self.out_whiten =  Whiten(num_groups=1,
                                  whitening_limit=_whitening_schedule(7.5),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.01)

    def forward(self,
                x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
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

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels,
                                     bias=True,
                                     initial_scale=0.05)



        self.whiten1 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)

        self.whiten2 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0, ratio=3.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)


    def forward(self,
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
        num_channels = x.shape[-1]
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.

        s = self.balancer(s)
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
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        return x


class ConvolutionModule(nn.Module):
<<<<<<< HEAD
    """ConvolutionModule in Zipformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py
=======
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """
<<<<<<< HEAD

    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0, kernel_size

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # after pointwise_conv1 we put x through a gated linear unit (nn.functional.glu).
=======
    def __init__(
            self, channels: int, kernel_size: int, causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels, 2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
<<<<<<< HEAD
        self.deriv_balancer1 = ActivationBalancer(
            2 * channels,
            channel_dim=1,
            max_abs=10.0,
            min_positive=0.05,
            max_positive=1.0,
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

        self.deriv_balancer2 = ActivationBalancer(
            channels,
            channel_dim=1,
            min_positive=0.05,
            max_positive=1.0,
            max_abs=20.0,
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
=======
        self.balancer1 = Balancer(
            bottleneck_dim, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity() # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity() # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = ChunkCausalDepthwiseConv1d(
            channels=bottleneck_dim,
            kernel_size=kernel_size) if causal else nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            groups=bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2)

        self.balancer2 = Balancer(
            bottleneck_dim, channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim, channels, activation='SwooshR',
            dropout_p=0.0, initial_scale=0.05,
        )

    def forward(self,
                x: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                chunk_size: int = -1,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
<<<<<<< HEAD
               (batch, #time), contains bool in masked positions.
=======
               (batch, #time), contains True in masked positions.
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
<<<<<<< HEAD
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = (T-3)//2 - 2 == (T-7)//2

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
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, (T-7)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer2_channels:
            Number of channels in layer2
          layer3_channels:
            Number of channels in layer3
        """
        assert in_channels >= 7, in_channels
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
            ),
            ActivationBalancer(layer1_channels, channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            ActivationBalancer(layer2_channels, channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2),  # (time, freq)
            ),
            ActivationBalancer(layer3_channels, channel_dim=1),
            DoubleSwish(),
        )
        out_height = (((in_channels - 1) // 2) - 1) // 2
        self.out = ScaledLinear(out_height * layer3_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, (T-7)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, (T-7)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        # Now x is of shape (N, (T-7)//2, odim)
        x = self.dropout(x)
        return x


class AttentionCombine(nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.

    All but the last input will have a linear transform before we
    randomly combine them; these linear transforms will be initialized
    to the identity transform.

    The idea is that the list of Tensors will be a list of outputs of multiple
    zipformer layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        num_channels: int,
        num_inputs: int,
        random_prob: float = 0.25,
        single_prob: float = 0.333,
    ) -> None:
        """
        Args:
          num_channels:
            the number of channels
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          random_prob:
            the probability with which we apply a nontrivial mask, in training
            mode.
         single_prob:
            the probability with which we mask to allow just a single
            module's output (in training)
        """
        super().__init__()

        self.random_prob = random_prob
        self.single_prob = single_prob
        self.weight = torch.nn.Parameter(torch.zeros(num_channels, num_inputs))
        self.bias = torch.nn.Parameter(torch.zeros(num_inputs))

        assert 0 <= random_prob <= 1, random_prob
        assert 0 <= single_prob <= 1, single_prob

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels).  In test mode
          this is just the final input.
        """
        num_inputs = self.weight.shape[1]
        assert len(inputs) == num_inputs

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        scores = (stacked_inputs * self.weight).sum(dim=(1,)) + self.bias

        if random.random() < 0.002:
            logging.info(f"Average scores are {scores.softmax(dim=1).mean(dim=0)}")

        if self.training:
            # random masking..
            mask_start = torch.randint(
                low=1,
                high=int(num_inputs / self.random_prob),
                size=(num_frames,),
                device=scores.device,
            ).unsqueeze(1)
            # mask will have rows like: [ False, False, False, True, True, .. ]
            arange = (
                torch.arange(num_inputs, device=scores.device)
                .unsqueeze(0)
                .expand(num_frames, num_inputs)
            )
            mask = arange >= mask_start

            apply_single_prob = torch.logical_and(
                torch.rand(size=(num_frames, 1), device=scores.device)
                < self.single_prob,
                mask_start < num_inputs,
            )
            single_prob_mask = torch.logical_and(
                apply_single_prob, arange < mask_start - 1
            )

            mask = torch.logical_or(mask, single_prob_mask)

            scores = scores.masked_fill(mask, float("-inf"))

        if self.training and random.random() < 0.1:
            scores = penalize_abs_values_gt(scores, limit=10.0, penalty=1.0e-04)

        weights = scores.softmax(dim=1)

        # (num_frames, num_channels, num_inputs) * (num_frames, num_inputs, 1) -> (num_frames, num_channels, 1),
        ans = torch.matmul(stacked_inputs, weights.unsqueeze(2))
        # ans: (*, num_channels)
        ans = ans.reshape(*tuple(inputs[0].shape[:-1]), num_channels)

        if __name__ == "__main__":
            # for testing only...
            print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans


def _test_random_combine():
    print("_test_random_combine()")
    num_inputs = 3
    num_channels = 50
    m = AttentionCombine(
        num_channels=num_channels,
        num_inputs=num_inputs,
        random_prob=0.5,
        single_prob=0.0,
    )

    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_zipformer_main():
    feature_dim = 50
    batch_size = 5
    seq_len = 20
    feature_dim = 50
    # Just make sure the forward pass runs.

    c = Zipformer(
        num_features=feature_dim,
        encoder_dims=(64, 96),
        encoder_unmasked_dims=(48, 64),
        nhead=(4, 4),
=======

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=-1)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if chunk_size >= 0:
            assert self.causal, "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1) # (time, batch, channels)

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
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    memory_dim = 100

    c = Zipformer2(
        encoder_dim=(64, 96), encoder_unmasked_dim=(48, 64), num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
        memory_dim=memory_dim,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
<<<<<<< HEAD
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    assert ((seq_len - 7) // 2 + 1) // 2 == f[0].shape[1], (seq_len, f.shape[1])
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
=======
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        memory=torch.randn(101, batch_size, memory_dim),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(seq_len, batch_size, 64),
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


<<<<<<< HEAD
def _test_conv2d_subsampling():
    num_features = 80
    encoder_dims = 384
    dropout = 0.1
    encoder_embed = Conv2dSubsampling(num_features, encoder_dims, dropout=dropout)
    for i in range(20, 40):
        x = torch.rand(2, i, num_features)
        y = encoder_embed(x)
        assert (x.shape[1] - 7) // 2 == y.shape[1], (x.shape[1], y.shape[1])


=======
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
<<<<<<< HEAD
    _test_random_combine()
    _test_zipformer_main()
    _test_conv2d_subsampling()
=======
    _test_zipformer_main(False)
    _test_zipformer_main(True)
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
