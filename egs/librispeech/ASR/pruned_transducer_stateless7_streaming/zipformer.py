#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey,)
#                                                  Zengwei Yao)
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

from icefall.utils import make_pad_mask, subsequent_chunk_mask


def stack_states(state_list: List[List[Tensor]]) -> List[Tensor]:
    """Stack list of zipformer states that correspond to separate utterances
    into a single emformer state, so that it can be used as an input for
    zipformer when those utterances are formed into a batch.

    Note:
      It is the inverse of :func:`unstack_states`.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance.
        ``states[i]`` is a list of 7 * num_encoders elements of i-th utterance.
        ``states[i][0:num_encoders]`` is the cached numbers of past frames.
        ``states[i][num_encoders:2*num_encoders]`` is the cached average tensors.
        ``states[i][2*num_encoders:3*num_encoders]`` is the cached key tensors of the first attention modules.
        ``states[i][3*num_encoders:4*num_encoders]`` is the cached value tensors of the first attention modules.
        ``states[i][4*num_encoders:5*num_encoders]`` is the cached value tensors of the second attention modules.
        ``states[i][5*num_encoders:6*num_encoders]`` is the cached left contexts of the first convolution modules.
        ``states[i][6*num_encoders:7*num_encoders]`` is the cached left contexts of the second convolution modules.

    Returns:
      A new state corresponding to a batch of utterances.
      See the input argument of :func:`unstack_states` for the meaning
      of the returned tensor.
    """
    batch_size = len(state_list)
    assert len(state_list[0]) % 7 == 0, len(state_list[0])
    num_encoders = len(state_list[0]) // 7

    cached_len = []
    cached_avg = []
    cached_key = []
    cached_val = []
    cached_val2 = []
    cached_conv1 = []
    cached_conv2 = []

    # For cached_len
    len_list = [state_list[n][0:num_encoders] for n in range(batch_size)]
    for i in range(num_encoders):
        # len_avg: (num_layers, batch_size)
        len_avg = torch.cat([len_list[n][i] for n in range(batch_size)], dim=1)
        cached_len.append(len_avg)

    # For cached_avg
    avg_list = [
        state_list[n][num_encoders : 2 * num_encoders] for n in range(batch_size)
    ]
    for i in range(num_encoders):
        # avg: (num_layers, batch_size, D)
        avg = torch.cat([avg_list[n][i] for n in range(batch_size)], dim=1)
        cached_avg.append(avg)

    # For cached_key
    key_list = [
        state_list[n][2 * num_encoders : 3 * num_encoders] for n in range(batch_size)
    ]
    for i in range(num_encoders):
        # key: (num_layers, left_context_size, batch_size, D)
        key = torch.cat([key_list[n][i] for n in range(batch_size)], dim=2)
        cached_key.append(key)

    # For cached_val
    val_list = [
        state_list[n][3 * num_encoders : 4 * num_encoders] for n in range(batch_size)
    ]
    for i in range(num_encoders):
        # val: (num_layers, left_context_size, batch_size, D)
        val = torch.cat([val_list[n][i] for n in range(batch_size)], dim=2)
        cached_val.append(val)

    # For cached_val2
    val2_list = [
        state_list[n][4 * num_encoders : 5 * num_encoders] for n in range(batch_size)
    ]
    for i in range(num_encoders):
        # val2: (num_layers, left_context_size, batch_size, D)
        val2 = torch.cat([val2_list[n][i] for n in range(batch_size)], dim=2)
        cached_val2.append(val2)

    # For cached_conv1
    conv1_list = [
        state_list[n][5 * num_encoders : 6 * num_encoders] for n in range(batch_size)
    ]
    for i in range(num_encoders):
        # conv1: (num_layers, batch_size, D, kernel-1)
        conv1 = torch.cat([conv1_list[n][i] for n in range(batch_size)], dim=1)
        cached_conv1.append(conv1)

    # For cached_conv2
    conv2_list = [
        state_list[n][6 * num_encoders : 7 * num_encoders] for n in range(batch_size)
    ]
    for i in range(num_encoders):
        # conv2: (num_layers, batch_size, D, kernel-1)
        conv2 = torch.cat([conv2_list[n][i] for n in range(batch_size)], dim=1)
        cached_conv2.append(conv2)

    states = (
        cached_len
        + cached_avg
        + cached_key
        + cached_val
        + cached_val2
        + cached_conv1
        + cached_conv2
    )
    return states


def unstack_states(states: List[Tensor]) -> List[List[Tensor]]:
    """Unstack the zipformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Note:
      It is the inverse of :func:`stack_states`.

    Args:
      states:
        A list of 7 * num_encoders elements:
        ``states[0:num_encoders]`` is the cached numbers of past frames.
        ``states[num_encoders:2*num_encoders]`` is the cached average tensors.
        ``states[2*num_encoders:3*num_encoders]`` is the cached key tensors of the first attention modules.
        ``states[3*num_encoders:4*num_encoders]`` is the cached value tensors of the first attention modules.
        ``states[4*num_encoders:5*num_encoders]`` is the cached value tensors of the second attention modules.
        ``states[5*num_encoders:6*num_encoders]`` is the cached left contexts of the first convolution modules.
        ``states[6*num_encoders:7*num_encoders]`` is the cached left contexts of the second convolution modules.

    Returns:
      A list of states.
      ``states[i]`` is a list of 7 * num_encoders elements of i-th utterance.
    """
    assert len(states) % 7 == 0, len(states)
    num_encoders = len(states) // 7
    (
        cached_len,
        cached_avg,
        cached_key,
        cached_val,
        cached_val2,
        cached_conv1,
        cached_conv2,
    ) = (states[i * num_encoders : (i + 1) * num_encoders] for i in range(7))

    batch_size = cached_len[0].shape[1]

    len_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_len[i]: (num_layers, batch_size)
        len_avg = cached_len[i].chunk(chunks=batch_size, dim=1)
        for n in range(batch_size):
            len_list[n].append(len_avg[n])

    avg_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_avg[i]: (num_layers, batch_size, D)
        avg = cached_avg[i].chunk(chunks=batch_size, dim=1)
        for n in range(batch_size):
            avg_list[n].append(avg[n])

    key_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_key[i]: (num_layers, left_context, batch_size, D)
        key = cached_key[i].chunk(chunks=batch_size, dim=2)
        for n in range(batch_size):
            key_list[n].append(key[n])

    val_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_val[i]: (num_layers, left_context, batch_size, D)
        val = cached_val[i].chunk(chunks=batch_size, dim=2)
        for n in range(batch_size):
            val_list[n].append(val[n])

    val2_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_val2[i]: (num_layers, left_context, batch_size, D)
        val2 = cached_val2[i].chunk(chunks=batch_size, dim=2)
        for n in range(batch_size):
            val2_list[n].append(val2[n])

    conv1_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_conv1[i]: (num_layers, batch_size, D, kernel-1)
        conv1 = cached_conv1[i].chunk(chunks=batch_size, dim=1)
        for n in range(batch_size):
            conv1_list[n].append(conv1[n])

    conv2_list = [[] for _ in range(batch_size)]
    for i in range(num_encoders):
        # cached_conv2[i]: (num_layers, batch_size, D, kernel-1)
        conv2 = cached_conv2[i].chunk(chunks=batch_size, dim=1)
        for n in range(batch_size):
            conv2_list[n].append(conv2[n])

    state_list = [
        (
            len_list[i]
            + avg_list[i]
            + key_list[i]
            + val_list[i]
            + val2_list[i]
            + conv1_list[i]
            + conv2_list[i]
        )
        for i in range(batch_size)
    ]
    return state_list


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
        cnn_module_kernels (int): Kernel size of convolution module
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
        num_left_chunks: int = 4,
        short_chunk_threshold: float = 0.75,
        short_chunk_size: int = 50,
        decode_chunk_size: int = 16,
        warmup_batches: float = 4000.0,
    ) -> None:
        super(Zipformer, self).__init__()

        self.num_features = num_features
        assert 0 < encoder_dims[0] <= encoder_dims[1]
        self.encoder_dims = encoder_dims
        self.encoder_unmasked_dims = encoder_unmasked_dims
        self.zipformer_downsampling_factors = zipformer_downsampling_factors
        self.output_downsampling_factor = output_downsampling_factor

        self.num_left_chunks = num_left_chunks
        self.short_chunk_threshold = short_chunk_threshold
        self.short_chunk_size = short_chunk_size

        # Used in decoding
        self.decode_chunk_size = decode_chunk_size

        self.left_context_len = self.decode_chunk_size * self.num_left_chunks

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

        self.num_encoder_layers = num_encoder_layers
        self.num_encoders = len(encoder_dims)
        self.attention_dims = attention_dim
        self.cnn_module_kernels = cnn_module_kernels
        for i in range(self.num_encoders):
            encoder_layer = ZipformerEncoderLayer(
                encoder_dims[i],
                attention_dim[i],
                nhead[i],
                feedforward_dim[i],
                dropout,
                cnn_module_kernels[i],
                pos_dim,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = ZipformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                dropout,
                warmup_begin=warmup_batches * (i + 1) / (self.num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (self.num_encoders + 1),
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
        If self.zipformer_downsampling_factors = (1, 2, 4, 8, 4, 2), then at the input of layer
        indexed 4 (in zero indexing), which has subsampling_factor=4, we combine the output of
        layers 2 and 3; and at the input of layer indexed 5, which has subsampling_factor=2,
        we combine the outputs of layers 1 and 4.
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
        if torch.jit.is_scripting() or not self.training:
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
            feature_masks.append(feature_mask)

        return feature_masks

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          chunk_size:
            The chunk size used in evaluation mode.
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

        if self.training:
            # Training mode
            max_ds = max(self.zipformer_downsampling_factors)
            # Generate dynamic chunk-wise attention mask during training
            max_len = x.size(0) // max_ds
            short_chunk_size = self.short_chunk_size // max_ds
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > (max_len * self.short_chunk_threshold):
                # Full attention
                chunk_size = x.size(0)
            else:
                # Chunk-wise attention
                chunk_size = chunk_size % short_chunk_size + 1
                chunk_size *= max_ds
        else:
            chunk_size = self.decode_chunk_size
            # Evaluation mode
            for ds in self.zipformer_downsampling_factors:
                assert chunk_size % ds == 0, (chunk_size, ds)

        attn_mask = ~subsequent_chunk_mask(
            size=x.size(0),
            chunk_size=chunk_size,
            num_left_chunks=self.num_left_chunks,
            device=x.device,
        )

        for i, (module, skip_module) in enumerate(
            zip(self.encoders, self.skip_modules)
        ):
            ds = self.zipformer_downsampling_factors[i]
            k = self.skip_layers[i]
            if isinstance(k, int):
                layer_skip_dropout_prob = self._get_layer_skip_dropout_prob()
                if torch.jit.is_scripting():
                    x = skip_module(outputs[k], x)
                elif (not self.training) or random.random() > layer_skip_dropout_prob:
                    x = skip_module(outputs[k], x)
            x = module(
                x,
                feature_mask=feature_masks[i],
                src_key_padding_mask=None if mask is None else mask[..., ::ds],
                attn_mask=attn_mask[::ds, ::ds],
            )
            outputs.append(x)

        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        lengths = (lengths + 1) >> 1

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: List[Tensor],
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
            seq_len is the input chunk length.
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states:
            A list of 7 * num_encoders elements:
            ``states[0:num_encoders]`` is the cached numbers of past frames.
            ``states[num_encoders:2*num_encoders]`` is the cached average tensors.
            ``states[2*num_encoders:3*num_encoders]`` is the cached key tensors of the first attention modules.
            ``states[3*num_encoders:4*num_encoders]`` is the cached value tensors of the first attention modules.
            ``states[4*num_encoders:5*num_encoders]`` is the cached value tensors of the second attention modules.
            ``states[5*num_encoders:6*num_encoders]`` is the cached left contexts of the first convolution modules.
            ``states[6*num_encoders:7*num_encoders]`` is the cached left contexts of the second convolution modules.

        Returns:
          Return a tuple containing 3 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, encoder_dims[-1])
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
            - updated states.
        """
        assert len(states) == 7 * self.num_encoders, (len(states), self.num_encoders)

        cached_len = states[: self.num_encoders]
        cached_avg = states[self.num_encoders : 2 * self.num_encoders]
        cached_key = states[2 * self.num_encoders : 3 * self.num_encoders]
        cached_val = states[3 * self.num_encoders : 4 * self.num_encoders]
        cached_val2 = states[4 * self.num_encoders : 5 * self.num_encoders]
        cached_conv1 = states[5 * self.num_encoders : 6 * self.num_encoders]
        cached_conv2 = states[6 * self.num_encoders : 7 * self.num_encoders]

        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        lengths = (x_lens - 7) >> 1
        assert x.size(0) == lengths.max().item(), (x.shape, lengths, lengths.max())

        outputs = []
        new_cached_len = []
        new_cached_avg = []
        new_cached_key = []
        new_cached_val = []
        new_cached_val2 = []
        new_cached_conv1 = []
        new_cached_conv2 = []

        for i, (module, skip_module) in enumerate(
            zip(self.encoders, self.skip_modules)
        ):
            k = self.skip_layers[i]
            if isinstance(k, int):
                x = skip_module(outputs[k], x)
            x, len_avg, avg, key, val, val2, conv1, conv2 = module.streaming_forward(
                x,
                cached_len=cached_len[i],
                cached_avg=cached_avg[i],
                cached_key=cached_key[i],
                cached_val=cached_val[i],
                cached_val2=cached_val2[i],
                cached_conv1=cached_conv1[i],
                cached_conv2=cached_conv2[i],
            )
            outputs.append(x)
            # Update caches
            new_cached_len.append(len_avg)
            new_cached_avg.append(avg)
            new_cached_key.append(key)
            new_cached_val.append(val)
            new_cached_val2.append(val2)
            new_cached_conv1.append(conv1)
            new_cached_conv2.append(conv2)

        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        lengths = (lengths + 1) >> 1

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        new_states = (
            new_cached_len
            + new_cached_avg
            + new_cached_key
            + new_cached_val
            + new_cached_val2
            + new_cached_conv1
            + new_cached_conv2
        )
        return x, lengths, new_states

    @torch.jit.export
    def get_init_state(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Get initial states.
        A list of 7 * num_encoders elements:
        ``states[0:num_encoders]`` is the cached numbers of past frames.
        ``states[num_encoders:2*num_encoders]`` is the cached average tensors.
        ``states[2*num_encoders:3*num_encoders]`` is the cached key tensors of the first attention modules.
        ``states[3*num_encoders:4*num_encoders]`` is the cached value tensors of the first attention modules.
        ``states[4*num_encoders:5*num_encoders]`` is the cached value tensors of the second attention modules.
        ``states[5*num_encoders:6*num_encoders]`` is the cached left contexts of the first convolution modules.
        ``states[6*num_encoders:7*num_encoders]`` is the cached left contexts of the second convolution modules.
        """
        cached_len = []
        cached_avg = []
        cached_key = []
        cached_val = []
        cached_val2 = []
        cached_conv1 = []
        cached_conv2 = []

        left_context_len = self.decode_chunk_size * self.num_left_chunks

        for i, encoder in enumerate(self.encoders):
            num_layers = encoder.num_layers
            ds = self.zipformer_downsampling_factors[i]

            len_avg = torch.zeros(num_layers, 1, dtype=torch.int64, device=device)
            cached_len.append(len_avg)

            avg = torch.zeros(num_layers, 1, encoder.d_model, device=device)
            cached_avg.append(avg)

            key = torch.zeros(
                num_layers,
                left_context_len // ds,
                1,
                encoder.attention_dim,
                device=device,
            )
            cached_key.append(key)

            val = torch.zeros(
                num_layers,
                left_context_len // ds,
                1,
                encoder.attention_dim // 2,
                device=device,
            )
            cached_val.append(val)

            val2 = torch.zeros(
                num_layers,
                left_context_len // ds,
                1,
                encoder.attention_dim // 2,
                device=device,
            )
            cached_val2.append(val2)

            conv1 = torch.zeros(
                num_layers,
                1,
                encoder.d_model,
                encoder.cnn_module_kernel - 1,
                device=device,
            )
            cached_conv1.append(conv1)

            conv2 = torch.zeros(
                num_layers,
                1,
                encoder.d_model,
                encoder.cnn_module_kernel - 1,
                device=device,
            )
            cached_conv2.append(conv2)

        states = (
            cached_len
            + cached_avg
            + cached_key
            + cached_val
            + cached_val2
            + cached_conv1
            + cached_conv2
        )
        return states


class ZipformerEncoderLayer(nn.Module):
    """
    ZipformerEncoderLayer is made up of self-attn, feedforward and convolution networks.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

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
        self.attention_dim = attention_dim
        self.cnn_module_kernel = cnn_module_kernel

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
        if torch.jit.is_scripting() or not self.training:
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
        if torch.jit.is_scripting() or not self.training:
            return 0.0
        warmup_period = 2000.0
        initial_dropout_rate = 0.2
        final_dropout_rate = 0.0
        if self.batch_count > warmup_period:
            return final_dropout_rate
        else:
            return initial_dropout_rate - (
                initial_dropout_rate - final_dropout_rate
            ) * (self.batch_count / warmup_period)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
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
        if torch.jit.is_scripting():
            src = src + self.pooling(src, src_key_padding_mask=src_key_padding_mask)
        elif random.random() >= dynamic_dropout:
            src = src + self.pooling(src, src_key_padding_mask=src_key_padding_mask)

        if torch.jit.is_scripting():
            src_att, attn_weights = self.self_attn(
                src,
                pos_emb=pos_emb,
                attn_mask=attn_mask,
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
                    attn_mask=attn_mask,
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

    def streaming_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        cached_len: Tensor,
        cached_avg: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            cached_len: processed number of past frames.
            cached_avg: cached average of past frames.
            cached_key: cached key tensor of left context for the first attention module.
            cached_val: cached value tensor of left context for the first attention module.
            cached_val2: cached value tensor of left context for the second attention module.
            cached_conv1: cached left context for the first convolution module.
            cached_conv2: cached left context for the second convolution module.

        Shape:
            src: (S, N, E).
            pos_emb: (N, left_context_len+2*S-1, E)
            cached_len: (N,)
              N is the batch size.
            cached_avg: (N, C).
              N is the batch size, C is the feature dimension.
            cached_key: (left_context_len, N, K).
              N is the batch size, K is the key dimension.
            cached_val: (left_context_len, N, V).
              N is the batch size, V is the key dimension.
            cached_val2: (left_context_len, N, V).
              N is the batch size, V is the key dimension.
            cached_conv1: (N, C, kernel_size-1).
              N is the batch size, C is the convolution channels.
            cached_conv2: (N, C, kernel_size-1).
              N is the batch size, C is the convolution channels.
        """
        src_orig = src

        # macaron style feed forward module
        src = src + self.feed_forward1(src)

        src_pool, cached_len, cached_avg = self.pooling.streaming_forward(
            src,
            cached_len=cached_len,
            cached_avg=cached_avg,
        )
        src = src + src_pool

        (
            src_attn,
            attn_weights,
            cached_key,
            cached_val,
        ) = self.self_attn.streaming_forward(
            src,
            pos_emb=pos_emb,
            cached_key=cached_key,
            cached_val=cached_val,
        )
        src = src + src_attn

        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src,
            cache=cached_conv1,
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)

        src_attn, cached_val2 = self.self_attn.streaming_forward2(
            src,
            attn_weights,
            cached_val=cached_val2,
        )
        src = src + src_attn

        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src,
            cache=cached_conv2,
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)

        src = self.norm_final(self.balancer(src))

        delta = src - src_orig

        src = src_orig + delta * self.bypass_scale

        return (
            src,
            cached_len,
            cached_avg,
            cached_key,
            cached_val,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


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

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        self.d_model = encoder_layer.d_model
        self.attention_dim = encoder_layer.attention_dim
        self.cnn_module_kernel = encoder_layer.cnn_module_kernel

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

    def forward(
        self,
        src: Tensor,
        # Note: The type of feature_mask should be Union[float, Tensor],
        # but to make torch.jit.script() work, we use `float` here
        feature_mask: float = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
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
        """
        pos_emb = self.encoder_pos(src)
        output = src

        if torch.jit.is_scripting():
            layers_to_drop = []
        else:
            rnd_seed = src.numel() + random.randint(0, 1000)
            layers_to_drop = self.get_layers_to_drop(rnd_seed)

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
            if not torch.jit.is_scripting():
                if i in layers_to_drop:
                    continue
            output = mod(
                output,
                pos_emb,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            output = output * feature_mask

        return output

    @torch.jit.export
    def streaming_forward(
        self,
        src: Tensor,
        cached_len: Tensor,
        cached_avg: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            cached_len: number of past frames.
            cached_avg: cached average of past frames.
            cached_key: cached key tensor for first attention module.
            cached_val: cached value tensor for first attention module.
            cached_val2: cached value tensor for second attention module.
            cached_conv1: cached left contexts for the first convolution module.
            cached_conv2: cached left contexts for the second convolution module.

        Shape:
            src: (S, N, E).
            cached_len: (num_layers,)
            cached_avg: (num_layers, N, C).
              N is the batch size, C is the feature dimension.
            cached_key: (num_layers, left_context_len, N, K).
              N is the batch size, K is the key dimension.
            cached_val: (num_layers, left_context_len, N, V).
              N is the batch size, V is the key dimension.
            cached_val2: (num_layers, left_context_len, N, V).
              N is the batch size, V is the key dimension.
            cached_conv1: (num_layers, N, C, kernel_size-1).
              N is the batch size, C is the convolution channels.
            cached_conv2: (num_layers, N, C, kernel_size-1).
              N is the batch size, C is the convolution channels.

        Returns: A tuple of 8 tensors:
            - output tensor
            - updated cached number of past frames.
            - updated cached average of past frames.
            - updated cached key tensor of of the first attention module.
            - updated cached value tensor of of the first attention module.
            - updated cached value tensor of of the second attention module.
            - updated cached left contexts of the first convolution module.
            - updated cached left contexts of the second convolution module.
        """
        assert cached_len.size(0) == self.num_layers, (
            cached_len.size(0),
            self.num_layers,
        )
        assert cached_avg.size(0) == self.num_layers, (
            cached_avg.size(0),
            self.num_layers,
        )
        assert cached_key.size(0) == self.num_layers, (
            cached_key.size(0),
            self.num_layers,
        )
        assert cached_val.size(0) == self.num_layers, (
            cached_val.size(0),
            self.num_layers,
        )
        assert cached_val2.size(0) == self.num_layers, (
            cached_val2.size(0),
            self.num_layers,
        )
        assert cached_conv1.size(0) == self.num_layers, (
            cached_conv1.size(0),
            self.num_layers,
        )
        assert cached_conv2.size(0) == self.num_layers, (
            cached_conv2.size(0),
            self.num_layers,
        )

        left_context_len = cached_key.shape[1]
        pos_emb = self.encoder_pos(src, left_context_len)
        output = src

        new_cached_len = []
        new_cached_avg = []
        new_cached_key = []
        new_cached_val = []
        new_cached_val2 = []
        new_cached_conv1 = []
        new_cached_conv2 = []
        for i, mod in enumerate(self.layers):
            output, len_avg, avg, key, val, val2, conv1, conv2 = mod.streaming_forward(
                output,
                pos_emb,
                cached_len=cached_len[i],
                cached_avg=cached_avg[i],
                cached_key=cached_key[i],
                cached_val=cached_val[i],
                cached_val2=cached_val2[i],
                cached_conv1=cached_conv1[i],
                cached_conv2=cached_conv2[i],
            )
            # Update caches
            new_cached_len.append(len_avg)
            new_cached_avg.append(avg)
            new_cached_key.append(key)
            new_cached_val.append(val)
            new_cached_val2.append(val2)
            new_cached_conv1.append(conv1)
            new_cached_conv2.append(conv2)

        return (
            output,
            torch.stack(new_cached_len, dim=0),
            torch.stack(new_cached_avg, dim=0),
            torch.stack(new_cached_key, dim=0),
            torch.stack(new_cached_val, dim=0),
            torch.stack(new_cached_val2, dim=0),
            torch.stack(new_cached_conv1, dim=0),
            torch.stack(new_cached_conv2, dim=0),
        )


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
        self.num_layers = encoder.num_layers
        self.d_model = encoder.d_model
        self.attention_dim = encoder.attention_dim
        self.cnn_module_kernel = encoder.cnn_module_kernel
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
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            attn_mask: attention mask (optional). Should be downsampled already.
            src_key_padding_mask: the mask for the src keys per batch (optional).  Should be downsampled already.

        Shape:
            src: (S, N, E).
            attn_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
        """
        src_orig = src
        src = self.downsample(src)

        src = self.encoder(
            src,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: Tensor,
        cached_len: Tensor,
        cached_avg: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            cached_avg: cached average value of past frames.
            cached_len: length of past frames.
            cached_key: cached key tensor for the first attention module.
            cached_val: cached value tensor for the first attention module.
            cached_val2: cached value tensor for the second attention module.
            cached_conv1: cached left context for the first convolution module.
            cached_conv2: cached left context for the second convolution module.

        Shape:
            src: (S, N, E).
            cached_len: (N,)
              N is the batch size.
            cached_avg: (num_layers, N, C).
              N is the batch size, C is the feature dimension.
            cached_key: (num_layers, left_context_len, N, K).
              N is the batch size, K is the key dimension.
            cached_val: (num_layers, left_context_len, N, V).
              N is the batch size, V is the key dimension.
            cached_val2: (num_layers, left_context_len, N, V).
              N is the batch size, V is the key dimension.
            cached_conv1: (num_layers, N, C, kernel_size-1).
              N is the batch size, C is the convolution channels.
            cached_conv2: (num_layers, N, C, kernel_size-1).
              N is the batch size, C is the convolution channels.
        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
        """
        src_orig = src
        src = self.downsample(src)

        (
            src,
            cached_len,
            cached_avg,
            cached_key,
            cached_val,
            cached_val2,
            cached_conv1,
            cached_conv2,
        ) = self.encoder.streaming_forward(
            src,
            cached_len=cached_len,
            cached_avg=cached_avg,
            cached_key=cached_key,
            cached_val=cached_val,
            cached_val2=cached_val2,
            cached_conv1=cached_conv1,
            cached_conv2=cached_conv2,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return (
            self.out_combiner(src_orig, src),
            cached_len,
            cached_avg,
            cached_key,
            cached_val,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


class AttentionDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: int):
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
        x: (seq_len, 1, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
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
        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.bias = nn.Parameter(torch.randn(upsample, num_channels) * 0.01)

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.bias.shape[0]
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src + self.bias.unsqueeze(1)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


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
        if not torch.jit.is_scripting():
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

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
    ) -> None:
        """Construct a PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        """Reset the positional encodings."""
        x_size_left = x.size(0) + left_context_len
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size_left * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size_left, self.d_model)
        pe_negative = torch.zeros(x_size_left, self.d_model)
        position = torch.arange(0, x_size_left, dtype=torch.float32).unsqueeze(1)
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

    def forward(self, x: torch.Tensor, left_context_len: int = 0) -> Tensor:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (time, batch, `*`).
            left_context_len: (int): Length of cached left context.

        Returns:
            torch.Tensor: Encoded tensor (batch, left_context_len + 2*time-1, `*`).

        """
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x_size_left
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(0),
        ]
        return self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
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

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
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

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x: input to be projected to query, key, value
            pos_emb: Positional embedding tensor

        Shape:
            - Inputs:
            - x: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - cached_key: :math:`(left_context_len, N, K)`, where N is the batch size, K is the key dimension.
            - cached_val: :math:`(left_context_len, N, V)`, where N is the batch size, V is the value dimension.

            - Returns: (attn_output, attn_weights, cached_key, cached_val)

              - attn_output: :math:`(S, N, E)` where S is the sequence length, N is the batch size,
                E is the embedding dimension.
              - attn_weights: :math:`(N * N, S, S)` where N is the batch size, H is the num-heads
                 and S is the sequence length.
              - cached_key: :math:`(left_context_len, N, K)`, updated cached attention key tensor of
                left context
              - cached_val: :math:`(left_context_len, N, K)`, updated cached attention value tensor of
        """
        (
            x,
            weights,
            cached_key,
            cached_val,
        ) = self.streaming_multi_head_attention_forward(
            self.in_proj(x),
            self.linear_pos(pos_emb),
            self.attention_dim,
            self.num_heads,
            self.out_proj.weight,
            self.out_proj.bias,
            cached_key=cached_key,
            cached_val=cached_val,
        )
        return x, weights, cached_key, cached_val

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

        if not torch.jit.is_scripting():
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

    def streaming_multi_head_attention_forward(
        self,
        x_proj: Tensor,
        pos: Tensor,
        attention_dim: int,
        num_heads: int,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x_proj: the projected input, to be split into query, key, value.
            pos: head-specific biases arising from the positional embeddings.
            attention_dim: dimension inside attention mechanism
            num_heads: parallel attention heads.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            cached_key: cached attention key tensor of left context.
            cached_val: cached attention value tensor of left context.

        Shape:
            Inputs:
            - x: :math:`(L, N, 7 * A // 2)` where L is the target sequence length, N is the batch size, A is
              the attention dimension.  Will be split into (query, key, value, pos).
            - pos: :math:`(N, 2*L-1, A//2)` or :math:`(1, 2*L-1, A//2)` where L is the sequence
              length, N is the batch size, and A is the attention dim.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_weights: :math:`(N * H, S, S)` where N is the batch size,
              H is the num-heads, S is the sequence length.
            - cached_key: :math:`(left_context_len, N, K)`, updated cached attention key tensor of left context.
            - cached_val: :math:`(left_context_len, N, K)`, updated cached attention value tensor of left context.
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

        left_context_len = cached_key.shape[0]
        assert left_context_len > 0, left_context_len
        assert cached_key.shape[0] == cached_val.shape[0], (
            cached_key.shape,
            cached_val.shape,
        )
        # Pad cached left contexts
        k = torch.cat([cached_key, k], dim=0)
        v = torch.cat([cached_val, v], dim=0)
        # Update cached left contexts
        cached_key = k[-left_context_len:, ...]
        cached_val = v[-left_context_len:, ...]

        # The length of key and value
        kv_len = k.shape[0]

        q = q.reshape(seq_len, bsz, num_heads, head_dim)
        p = p.reshape(seq_len, bsz, num_heads, pos_dim)
        k = k.reshape(kv_len, bsz, num_heads, head_dim)
        v = v.reshape(kv_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        q = q.permute(1, 2, 0, 3)  # (batch, head, time1, head_dim)
        p = p.permute(1, 2, 0, 3)  # (batch, head, time1, pos_dim)
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)

        seq_len2 = 2 * seq_len - 1 + left_context_len
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
            cols = torch.arange(kv_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_weights = pos_weights.reshape(-1, n)
            pos_weights = torch.gather(pos_weights, dim=1, index=indexes)
            pos_weights = pos_weights.reshape(batch_size, num_heads, time1, kv_len)
        else:
            pos_weights = pos_weights.as_strided(
                (bsz, num_heads, seq_len, kv_len),
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

        # attn_output_weights: (batch, head, time1, time2)
        attn_output_weights = attn_output_weights.view(bsz * num_heads, seq_len, kv_len)

        # Using this version of softmax, defined in scaling.py,
        # should save a little of the memory used in backprop by, if
        # we are in automatic mixed precision mode (amp) == autocast,
        # only storing the half-precision output for backprop purposes.
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, seq_len, head_dim // 2]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, attention_dim // 2)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        return attn_output, attn_output_weights, cached_key, cached_val

    def forward2(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
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

        if not torch.jit.is_scripting():
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

    def streaming_forward2(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Second forward function, where we re-use the attn_weights returned by the first forward function
        but with different input.
        Args:
            x: input, of shape (seq_len, batch_size, embed_dim)
            attn_weights: attention weights returned by forward(), of shape (batch_size * num_heads, seq_len, seq_len)
            cached_val: cached attention value tensor of left context.
        Returns:
            - output of the same shape as x, i.e. (seq_len, batch_size, embed_dim)
            - updated cached attention value tensor of left context.
        """
        num_heads = self.num_heads
        (seq_len, bsz, embed_dim) = x.shape
        head_dim = self.attention_dim // num_heads
        # v: (tgt_len, bsz, embed_dim // 2)
        v = self.in_proj2(x)

        left_context_len = cached_val.shape[0]
        assert left_context_len > 0, left_context_len
        v = torch.cat([cached_val, v], dim=0)
        cached_val = v[-left_context_len:]

        seq_len2 = left_context_len + seq_len
        v = v.reshape(seq_len2, bsz * num_heads, head_dim // 2).transpose(0, 1)

        # now v: (bsz * num_heads, seq_len, head_dim // 2)
        attn_output = torch.bmm(attn_weights, v)

        # attn_output: (bsz * num_heads, seq_len, head_dim)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, self.attention_dim // 2)
        )
        # returned value is of shape (seq_len, bsz, embed_dim), like x.
        return self.out_proj2(attn_output), cached_val

    def _print_attn_stats(self, attn_weights: Tensor, attn_output: Tensor):
        # attn_weights: (batch_size * num_heads, seq_len, seq_len)
        # attn_output: (bsz * num_heads, seq_len, head_dim)
        (n, seq_len, head_dim) = attn_output.shape
        num_heads = self.num_heads
        bsz = n // num_heads

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
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

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
           x: a Tensor of shape (T, N, C)
           src_key_padding_mask: a Tensor of bool, of shape (N, T), with True in masked
               positions.

        Returns:
           - output, a Tensor of shape (T, N, C).
        """
        if src_key_padding_mask is not None:
            # False in padding positions
            padding_mask = src_key_padding_mask.logical_not().to(x.dtype)  # (N, T)
            # Cumulated numbers of frames from start
            cum_mask = padding_mask.cumsum(dim=1)  # (N, T)
            x = x.cumsum(dim=0)  # (T, N, C)
            pooling_mask = padding_mask / cum_mask
            pooling_mask = pooling_mask.transpose(0, 1).contiguous().unsqueeze(-1)
            # now pooling_mask: (T, N, 1)
            x = x * pooling_mask  # (T, N, C)
        else:
            num_frames = x.shape[0]
            cum_mask = torch.arange(1, num_frames + 1).unsqueeze(1)  # (T, 1)
            x = x.cumsum(dim=0)  # (T, N, C)
            pooling_mask = (1.0 / cum_mask).unsqueeze(2)
            # now pooling_mask: (T, N, 1)
            x = x * pooling_mask

        x = self.proj(x)
        return x

    def streaming_forward(
        self,
        x: Tensor,
        cached_len: Tensor,
        cached_avg: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
           x: a Tensor of shape (T, N, C)
           cached_len: a Tensor of int, of shape (N,), containing the number of
               past frames in batch.
           cached_avg: a Tensor of shape (N, C), the average over all past frames
               in batch.

        Returns:
           A tuple of 2 tensors:
           - output, a Tensor of shape (T, N, C).
           - updated cached_avg, a Tensor of shape (N, C).
        """
        x = x.cumsum(dim=0)  # (T, N, C)
        x = x + (cached_avg * cached_len.unsqueeze(1)).unsqueeze(0)
        # Cumulated numbers of frames from start
        cum_mask = torch.arange(1, x.size(0) + 1, device=x.device)
        cum_mask = cum_mask.unsqueeze(1) + cached_len.unsqueeze(0)  # (T, N)
        pooling_mask = (1.0 / cum_mask).unsqueeze(2)
        # now pooling_mask: (T, N, 1)
        x = x * pooling_mask  # (T, N, C)

        cached_len = cached_len + x.size(0)
        cached_avg = x[-1]

        x = self.proj(x)
        return x, cached_len, cached_avg


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
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

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
        self.deriv_balancer1 = ActivationBalancer(
            2 * channels,
            channel_dim=1,
            max_abs=10.0,
            min_positive=0.05,
            max_positive=1.0,
        )

        # Will pad cached left context
        self.lorder = kernel_size - 1
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,
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
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains bool in masked positions.

        Returns:
            - Output tensor (#time, batch, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        # 1D Depthwise Conv
        # Make depthwise_conv causal by
        # manualy padding self.lorder zeros to the left
        x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for the src keys per batch:
               (batch, #time), contains bool in masked positions.
            cache: Cached left context for depthwise_conv, with shape of
               (batch, channels, #kernel_size-1). Only used in real streaming decoding.

        Returns:
            A tuple of 2 tensors:
            - Output tensor (#time, batch, channels).
            - New cached left context, with shape of (batch, channels, #kernel_size-1).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        assert cache.shape == (x.size(0), x.size(1), self.lorder), (
            cache.shape,
            (x.size(0), x.size(1), self.lorder),
        )
        x = torch.cat([cache, x], dim=2)
        # Update cache
        cache = x[:, :, -self.lorder :]
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1), cache


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


def _test_zipformer_main():
    feature_dim = 50
    batch_size = 5
    seq_len = 47
    feature_dim = 50
    # Just make sure the forward pass runs.

    c = Zipformer(
        num_features=feature_dim,
        encoder_dims=(64, 96),
        encoder_unmasked_dims=(48, 64),
        nhead=(4, 4),
        decode_chunk_size=4,
    )
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    assert ((seq_len - 7) // 2 + 1) // 2 == f[0].shape[1], (seq_len, f.shape[1])
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


def _test_conv2d_subsampling():
    num_features = 80
    encoder_dims = 384
    dropout = 0.1
    encoder_embed = Conv2dSubsampling(num_features, encoder_dims, dropout=dropout)
    for i in range(20, 40):
        x = torch.rand(2, i, num_features)
        y = encoder_embed(x)
        assert (x.shape[1] - 7) // 2 == y.shape[1], (x.shape[1], y.shape[1])


def _test_pooling_module():
    N, S, C = 2, 12, 32
    chunk_len = 4
    m = PoolingModule(d_model=C)

    # test chunk-wise forward with padding_mask
    x = torch.randn(S, N, C)
    y = m(x)
    cached_len = torch.zeros(N, dtype=torch.int32)
    cached_avg = torch.zeros(N, C)
    for i in range(S // chunk_len):
        start = i * chunk_len
        end = start + chunk_len
        x_chunk = x[start:end]
        y_chunk, cached_len, cached_avg = m.streaming_forward(
            x_chunk,
            cached_len=cached_len,
            cached_avg=cached_avg,
        )
        assert torch.allclose(y_chunk, y[start:end]), (y_chunk, y[start:end])


def _test_state_stack_unstack():
    m = Zipformer(
        num_features=80,
        encoder_dims=(64, 96),
        encoder_unmasked_dims=(48, 64),
        nhead=(4, 4),
        zipformer_downsampling_factors=(4, 8),
        num_left_chunks=2,
        decode_chunk_size=8,
    )
    s1 = m.get_init_state()
    s2 = m.get_init_state()
    states = stack_states([s1, s2])
    new_s1, new_s2 = unstack_states(states)
    for i in range(m.num_encoders * 7):
        for x, y in zip(s1[i], new_s1[i]):
            assert torch.equal(x, y)
        for x, y in zip(s2[i], new_s2[i]):
            assert torch.equal(x, y)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main()
    _test_conv2d_subsampling()
    _test_pooling_module()
    _test_state_stack_unstack()
