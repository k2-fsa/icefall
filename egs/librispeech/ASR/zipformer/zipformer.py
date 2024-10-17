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
)
from scaling import (
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
)
from scaling import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
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


class Zipformer2(torch.nn.Module):
    """
    Zipformer2 encoder.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        input_dim: int,
        subsample_output_dim: int,
        subsample_layer1_channels: int,
        subsample_layer2_channels: int,
        subsample_layer3_channels: int,
        encoder_dims: list[int],
        num_encoder_layers: list[int],
        downsampling_factors: list[int],
        num_heads: list[int],
        feedforward_dims: list[int],
        cnn_module_kernels: list[int],
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        pos_dim: int,
        pos_max_len: int,
        output_dim: int,
        use_ctc: bool,
        left_context_frames: int,
        right_context_frames: int,
        device: torch.device,
    ) -> None:
        """
        Zipformer2 initialization.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        subsample_output_dim : int
            The output dimension of the subsampling module represented by Conv2dSubsampling.
        subsample_layer1_channels : int
            The number of output channels in the first Conv2d layer of the
            Conv2dSubsampling module.
        subsample_layer2_channels : int
            The number of output channels in the second Conv2d layer of the
            Conv2dSubsampling module.
        subsample_layer3_channels : int
            The number of output channels in the third Conv2d layer of the
            Conv2dSubsampling module.
        encoder_dims : list[int]
            A list of 5 integers, the embedding dimension of
            Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        num_encoder_layers : list[int]
            A list of 5 integers, the number of Zipformer2EncoderLayer
            modules in each Zipformer2Encoder stack.
        downsampling_factors : list[int]
            A list of 5 integers, the downsampling factor of each Zipformer2Encoder stack.
            Note: this is in addition to the downsampling factor of 2 that is applied in the
            Conv2dSubsampling module.
        num_heads : list[int]
            A list of 5 integers, the number of heads for attention weights and self-attention of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        feedforward_dims : list[int]
            A list of 5 integers, the hidden dimension of the feedforward module of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        cnn_module_kernels : list[int]
            A list of 5 integers, the kernel size of the convolution module of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        query_head_dim : int
            The dimension of the query and key per attention head in attention weights of the
            Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        pos_head_dim : int
            The dimension of the projected positional encoding per attention head in attention
            weights of the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        value_head_dim : int
            The dimension of the value per attention head in self-attention of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        pos_dim: int
            The dimension of the relative positional embeddings in each Zipformer2Encoder stack.
        pos_max_len : int
            The maximum input duration of the relative positional embeddings in each
            Zipformer2Encoder stack. Note: if the input duration of any positional embedding module
            exceeds this number, then one might end up with a big degradation of inference speed.
        output_dim : int
            The output dimension after final output projection.
        use_ctc : bool
            If True, assuming that ctc head will loaded to the output encoder projection.
            In this case torch.nn.functional. will be applied to the output at the very end.
        left_context_frames : int
            The left context number of frames after the initial subsampling with
            Conv2dSubsampling module.
        right_context_frames : int
            The right (look-ahead) context number of frames.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """
        # pylint: disable=too-many-arguments,too-many-locals

        super().__init__()

        if not (
            len(encoder_dims)
            == len(num_encoder_layers)
            == len(downsampling_factors)
            == len(num_heads)
            == len(feedforward_dims)
            == len(cnn_module_kernels)
            == 6
        ):
            raise ValueError(
                'It is required that the length of encoder_dims, num_encoder_layers, '
                'downsampling_factors, num_heads, feedforward_dims, and cnn_module_kernels is the '
                'same and equal to 6, but got following list lengths:\n'
                f'len(num_encoder_layers) == {len(num_encoder_layers)}\n'
                f'len(downsampling_factors) == {len(downsampling_factors)}\n'
                f'len(encoder_dims) == {len(encoder_dims)}\n'
                f'len(num_heads) == {len(num_heads)}\n'
                f'len(cnn_module_kernels) == {len(cnn_module_kernels)}\n'
                f'len(feedforward_dims) == {len(feedforward_dims)}.',
            )

        self.encoder_dims = tuple(encoder_dims)
        self.downsampling_factors = tuple(downsampling_factors)
        self.left_context_frames = left_context_frames
        projection_dim = max(encoder_dims)
        self.projection_dim = projection_dim
        self.ctc = use_ctc

        self.subsampling = Conv2dSubsampling(
            input_dim,
            subsample_output_dim,
            subsample_layer1_channels,
            subsample_layer2_channels,
            subsample_layer3_channels,
            right_context_frames,
            device,
        )

        encoders = []
        for i, num_layers in enumerate(num_encoder_layers):

            encoder_layer = Zipformer2EncoderLayer(
                encoder_dims[i],
                pos_dim,
                num_heads[i],
                query_head_dim,
                pos_head_dim,
                value_head_dim,
                feedforward_dims[i],
                cnn_module_kernels[i],
                left_context_frames // downsampling_factors[i],
                right_context_frames // 2 // downsampling_factors[i],
                device,
            )

            encoder = Zipformer2Encoder(
                encoder_layer,
                num_layers,
                encoder_dims[i],
                pos_dim,
                pos_max_len,
                downsampling_factors[i],
                device,
            )

            encoders.append(encoder)

        self.encoder_1 = encoders[0]
        self.encoder_2 = encoders[1]
        self.encoder_3 = encoders[2]
        self.encoder_4 = encoders[3]
        self.encoder_5 = encoders[4]
        self.encoder_6 = encoders[5]

        self.downsample_output = SimpleDownsample(2, device)
        self.projection_output = torch.nn.Linear(projection_dim, output_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        # We need to preserve this explicit arguments reference for the smooth
        # TirchScript export with the following ONNX export.
        left_cached_subsample_frames: torch.Tensor,

        left_cached_keys_encoder_1: torch.Tensor,
        left_cached_nonlin_attentions_encoder_1: torch.Tensor,
        left_cached_values_1_encoder_1: torch.Tensor,
        left_cached_values_2_encoder_1: torch.Tensor,
        left_cached_convolutions_1_encoder_1: torch.Tensor,
        left_cached_convolutions_2_encoder_1: torch.Tensor,

        left_cached_keys_encoder_2: torch.Tensor,
        left_cached_nonlin_attentions_encoder_2: torch.Tensor,
        left_cached_values_1_encoder_2: torch.Tensor,
        left_cached_values_2_encoder_2: torch.Tensor,
        left_cached_convolutions_1_encoder_2: torch.Tensor,
        left_cached_convolutions_2_encoder_2: torch.Tensor,

        left_cached_keys_encoder_3: torch.Tensor,
        left_cached_nonlin_attentions_encoder_3: torch.Tensor,
        left_cached_values_1_encoder_3: torch.Tensor,
        left_cached_values_2_encoder_3: torch.Tensor,
        left_cached_convolutions_1_encoder_3: torch.Tensor,
        left_cached_convolutions_2_encoder_3: torch.Tensor,

        left_cached_keys_encoder_4: torch.Tensor,
        left_cached_nonlin_attentions_encoder_4: torch.Tensor,
        left_cached_values_1_encoder_4: torch.Tensor,
        left_cached_values_2_encoder_4: torch.Tensor,
        left_cached_convolutions_1_encoder_4: torch.Tensor,
        left_cached_convolutions_2_encoder_4: torch.Tensor,

        left_cached_keys_encoder_5: torch.Tensor,
        left_cached_nonlin_attentions_encoder_5: torch.Tensor,
        left_cached_values_1_encoder_5: torch.Tensor,
        left_cached_values_2_encoder_5: torch.Tensor,
        left_cached_convolutions_1_encoder_5: torch.Tensor,
        left_cached_convolutions_2_encoder_5: torch.Tensor,

        left_cached_keys_encoder_6: torch.Tensor,
        left_cached_nonlin_attentions_encoder_6: torch.Tensor,
        left_cached_values_1_encoder_6: torch.Tensor,
        left_cached_values_2_encoder_6: torch.Tensor,
        left_cached_convolutions_1_encoder_6: torch.Tensor,
        left_cached_convolutions_2_encoder_6: torch.Tensor,

        processed_len: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        Does a forward pass of the Zipformer2 module, which represents the whole acoustic encoder.
        Returns a tuple with the output tensor, updated left cache feature tensor for subsampling
        module, 36 left cache tensors for multiple attention and convolution modules within each of
        6 Zipformer2Encoder modules, and finally, the updated processed length single-element
        tensor with the total number of processed frames after subsampling module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float feature tensor of shape (1, num_frames, input_dim),
            where the input_dim corresponds to the number of features.
        left_cached_subsample_frames : torch.Tensor[torch.float32]
            The subsampling module left cache tensor of shape (1, 10, input_dim).
        left_cached_keys_encoder_1 : torch.Tensor[torch.float32]
            The cached attention key tensor of the left context of each
            Zipformer2EncoderLayer within the first Zipformer2Encoder.
            The tensor is of shape (num_layers_1, 1, left_context_len_1, query_dim_1).
        left_cached_nonlin_attentions_encoder_1 : torch.Tensor[torch.float32]
            The left context cached attention tensor for the non-linear attention module of each
            Zipformer2EncoderLayer within the first Zipformer2Encoder.
            The tensor is of shape (num_layers_1, 1, left_context_len_1, head_dim_1).
        left_cached_values_1_encoder_1 : torch.Tensor[torch.float32]
            The cached left context tensor for the first self-attention module of each
            Zipformer2EncoderLayer within the first Zipformer2Encoder.
            The tensor is of shape (num_layers_1, 1, left_context_len_1, value_dim_1).
        left_cached_values_2_encoder_1 : torch.Tensor[torch.float32]
            The cached left context tensor for the second self-attention module of each
            Zipformer2EncoderLayer within the first Zipformer2Encoder.
            The tensor is of shape (num_layers_1, 1, left_context_len_1, value_dim_1).
        left_cached_convolutions_1_encoder_1 : torch.Tensor[torch.float32]
            The cached left context tensor for the first convolution module of each
            Zipformer2EncoderLayer within the first Zipformer2Encoder.
            The tensor is of shape (num_layers_1, 1, embed_dim_1, left_cache_len_1).
        left_cached_convolutions_2_encoder_1 : torch.Tensor[torch.float32]
            The cached left context tensor for the second convolution module of each
            Zipformer2EncoderLayer within the first Zipformer2Encoder.
            The tensor is of shape (num_layers_1, 1, embed_dim_1, left_cache_len_1).
                                            .
                                            .
                                            .
        left_cached_convolutions_2_encoder_6 : torch.Tensor[torch.float32]
            The cached left context tensor for the second convolution module of each
            Zipformer2EncoderLayer within the sixth Zipformer2Encoder.
            The tensor is of shape (num_layers_6, 1, embed_dim_6, left_cache_len_6).
        processed_len : torch.Tensor[torch.int32]
            The total processed length after subsampling, single-element integer tensor
            of shape (1,).

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.float32],
            torch.Tensor[torch.float32], torch.Tensor[torch.float32], torch.Tensor[torch.int32],
        ]
            A tuple of 38 float tensors and 1 integer tensor:
            - The module output of shape (1, seq_len, output_dim).
            - The updated subsampling module left cache tensor of shape (1, 10, input_dim).
            - The updated cached attention key tensor of the left context of each
              Zipformer2EncoderLayer within the first Zipformer2Encoder.
              The tensor is of shape (num_layers_1, 1, left_context_len_1, query_dim_1).
            - The updated left context cached attention tensor for the non-linear attention
              module of each Zipformer2EncoderLayer within the first Zipformer2Encoder.
              The tensor is of shape (num_layers_1, 1, left_context_len_1, head_dim_1).
            - The updated cached left context tensor for the first self-attention module of each
              Zipformer2EncoderLayer within the first Zipformer2Encoder.
              The tensor is of shape (num_layers_1, 1, left_context_len_1, value_dim_1).
            - The updated cached left context tensor for the second
              self-attention module of each Zipformer2EncoderLayer within the first
              Zipformer2Encoder.
              The tensor is of shape (num_layers_1, 1, left_context_len_1, value_dim_1).
            - The updated cached left context tensor for the first convolution module of each
              Zipformer2EncoderLayer within the first Zipformer2Encoder.
              The tensor is of shape (num_layers_1, 1, embed_dim_1, left_cache_len_1).
            - The updated cached left context tensor for the second convolution module of each
              Zipformer2EncoderLayer within the first Zipformer2Encoder.
              The tensor is of shape (num_layers_1, 1, embed_dim_1, left_cache_len_1).
                                            .
                                            .
                                            .
            - The updated cached left context tensor for the second convolution module of each
              Zipformer2EncoderLayer within the sixth Zipformer2Encoder.
              The tensor is of shape (num_layers_6, 1, embed_dim_6, left_cache_len_6).
            - The updated total processed length tensor after subsampling of shape (1,).
        """
        # pylint: disable=too-many-arguments,too-many-locals

        x, new_left_cached_subsample_frames = self.subsampling(x, left_cached_subsample_frames)

        batch_size, seq_len, _ = x.size()
        src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)

        # processed_mask is used to mask out the initial self.states if left_context_frames == 6,
        # then tensor will contain [5, 4, 3, 2, 1, 0] as if reversed
        # torch.arange(left_context_frames).
        processed_mask = torch.arange(
            self.left_context_frames - 1, -1, -1, dtype=torch.int32, device=x.device,
        ).expand(batch_size, self.left_context_frames)

        # (1, left_context_size) i.e. (batch_size, left_context_size)
        processed_mask = processed_mask >= processed_len.expand(processed_mask.size())

        # Update processed lengths
        new_processed_len = processed_len + seq_len

        # (1, left_context_size + chunk_size)
        src_key_padding_mask = torch.cat((processed_mask, src_key_padding_mask), dim=1)

        # If the last encoder 'x' has the largest dimension, then the 'output' will be just this
        # last 'x' unchanged. Otherwise it will be concatenated from different pieces of 'x',
        # taking each output channel dimension from the most recent x that has it present.
        output = torch.empty(
            batch_size, seq_len, self.projection_dim, dtype=torch.float32, device=x.device,
        )

        # We have a number of Zipformer2Encoder stacks fixed and equal to 6 for any Ziformer2 size
        # including small, medium and large. For the sake of smoother model TorchScript export we
        # engage sequential explicit forward call of each Zipformer2Encoder module instead of using
        # torch.nn.ModuleList.

        # Encoder 1

        (
            x,
            new_left_cached_keys_encoder_1,
            new_left_cached_nonlin_attentions_encoder_1,
            new_left_cached_values_1_encoder_1,
            new_left_cached_values_2_encoder_1,
            new_left_cached_convolutions_1_encoder_1,
            new_left_cached_convolutions_2_encoder_1,
        ) = self.encoder_1(
            x,
            left_cached_keys_encoder_1,
            left_cached_nonlin_attentions_encoder_1,
            left_cached_values_1_encoder_1,
            left_cached_values_2_encoder_1,
            left_cached_convolutions_1_encoder_1,
            left_cached_convolutions_2_encoder_1,
            src_key_padding_mask[:, ::self.downsampling_factors[0]],
        )
        output[:, :, :x.size(2)] = x

        # Encoder 2

        pad = torch.zeros(
            x.size(0), x.size(1), self.encoder_dims[1] - x.size(2),
            dtype=torch.float32,
            device=x.device,
        )
        x = torch.cat((x, pad), dim=2)

        (
            x,
            new_left_cached_keys_encoder_2,
            new_left_cached_nonlin_attentions_encoder_2,
            new_left_cached_values_1_encoder_2,
            new_left_cached_values_2_encoder_2,
            new_left_cached_convolutions_1_encoder_2,
            new_left_cached_convolutions_2_encoder_2,
        ) = self.encoder_2(
            x,
            left_cached_keys_encoder_2,
            left_cached_nonlin_attentions_encoder_2,
            left_cached_values_1_encoder_2,
            left_cached_values_2_encoder_2,
            left_cached_convolutions_1_encoder_2,
            left_cached_convolutions_2_encoder_2,
            src_key_padding_mask[:, ::self.downsampling_factors[1]],
        )
        output[:, :, :x.size(2)] = x

        # Encoder 3

        pad = torch.zeros(
            x.size(0), x.size(1), self.encoder_dims[2] - x.size(2),
            dtype=torch.float32,
            device=x.device,
        )
        x = torch.cat((x, pad), dim=2)

        (
            x,
            new_left_cached_keys_encoder_3,
            new_left_cached_nonlin_attentions_encoder_3,
            new_left_cached_values_1_encoder_3,
            new_left_cached_values_2_encoder_3,
            new_left_cached_convolutions_1_encoder_3,
            new_left_cached_convolutions_2_encoder_3,
        ) = self.encoder_3(
            x,
            left_cached_keys_encoder_3,
            left_cached_nonlin_attentions_encoder_3,
            left_cached_values_1_encoder_3,
            left_cached_values_2_encoder_3,
            left_cached_convolutions_1_encoder_3,
            left_cached_convolutions_2_encoder_3,
            src_key_padding_mask[:, ::self.downsampling_factors[2]],
        )
        output[:, :, :x.size(2)] = x

        # Encoder 4

        pad = torch.zeros(
            x.size(0), x.size(1), self.encoder_dims[3] - x.size(2),
            dtype=torch.float32,
            device=x.device,
        )
        x = torch.cat((x, pad), dim=2)

        (
            x,
            new_left_cached_keys_encoder_4,
            new_left_cached_nonlin_attentions_encoder_4,
            new_left_cached_values_1_encoder_4,
            new_left_cached_values_2_encoder_4,
            new_left_cached_convolutions_1_encoder_4,
            new_left_cached_convolutions_2_encoder_4,
        ) = self.encoder_4(
            x,
            left_cached_keys_encoder_4,
            left_cached_nonlin_attentions_encoder_4,
            left_cached_values_1_encoder_4,
            left_cached_values_2_encoder_4,
            left_cached_convolutions_1_encoder_4,
            left_cached_convolutions_2_encoder_4,
            src_key_padding_mask[:, ::self.downsampling_factors[3]],
        )
        output[:, :, :x.size(2)] = x

        # Encoder 5

        x = x[:, :, :self.encoder_dims[4]]
        (
            x,
            new_left_cached_keys_encoder_5,
            new_left_cached_nonlin_attentions_encoder_5,
            new_left_cached_values_1_encoder_5,
            new_left_cached_values_2_encoder_5,
            new_left_cached_convolutions_1_encoder_5,
            new_left_cached_convolutions_2_encoder_5,
        ) = self.encoder_5(
            x,
            left_cached_keys_encoder_5,
            left_cached_nonlin_attentions_encoder_5,
            left_cached_values_1_encoder_5,
            left_cached_values_2_encoder_5,
            left_cached_convolutions_1_encoder_5,
            left_cached_convolutions_2_encoder_5,
            src_key_padding_mask[:, ::self.downsampling_factors[4]],
        )
        output[:, :, :x.size(2)] = x

        # Encoder 6

        x = x[:, :, :self.encoder_dims[5]]
        (
            x,
            new_left_cached_keys_encoder_6,
            new_left_cached_nonlin_attentions_encoder_6,
            new_left_cached_values_1_encoder_6,
            new_left_cached_values_2_encoder_6,
            new_left_cached_convolutions_1_encoder_6,
            new_left_cached_convolutions_2_encoder_6,
        ) = self.encoder_6(
            x,
            left_cached_keys_encoder_6,
            left_cached_nonlin_attentions_encoder_6,
            left_cached_values_1_encoder_6,
            left_cached_values_2_encoder_6,
            left_cached_convolutions_1_encoder_6,
            left_cached_convolutions_2_encoder_6,
            src_key_padding_mask[:, ::self.downsampling_factors[5]],
        )
        output[:, :, :x.size(2)] = x

        output = self.downsample_output(output)
        output = self.projection_output(output)
        if self.ctc:
            output = torch.nn.functional.log_softmax(output, dim=2)

        return (
            output,
            # Because of the reasons mentioned in previous comments,
            # for the sake of easier TorchScript and ONNX export we
            # preserve the explicit listing of each left cache tensor.
            new_left_cached_subsample_frames,

            new_left_cached_keys_encoder_1,
            new_left_cached_nonlin_attentions_encoder_1,
            new_left_cached_values_1_encoder_1,
            new_left_cached_values_2_encoder_1,
            new_left_cached_convolutions_1_encoder_1,
            new_left_cached_convolutions_2_encoder_1,

            new_left_cached_keys_encoder_2,
            new_left_cached_nonlin_attentions_encoder_2,
            new_left_cached_values_1_encoder_2,
            new_left_cached_values_2_encoder_2,
            new_left_cached_convolutions_1_encoder_2,
            new_left_cached_convolutions_2_encoder_2,

            new_left_cached_keys_encoder_3,
            new_left_cached_nonlin_attentions_encoder_3,
            new_left_cached_values_1_encoder_3,
            new_left_cached_values_2_encoder_3,
            new_left_cached_convolutions_1_encoder_3,
            new_left_cached_convolutions_2_encoder_3,

            new_left_cached_keys_encoder_4,
            new_left_cached_nonlin_attentions_encoder_4,
            new_left_cached_values_1_encoder_4,
            new_left_cached_values_2_encoder_4,
            new_left_cached_convolutions_1_encoder_4,
            new_left_cached_convolutions_2_encoder_4,

            new_left_cached_keys_encoder_5,
            new_left_cached_nonlin_attentions_encoder_5,
            new_left_cached_values_1_encoder_5,
            new_left_cached_values_2_encoder_5,
            new_left_cached_convolutions_1_encoder_5,
            new_left_cached_convolutions_2_encoder_5,

            new_left_cached_keys_encoder_6,
            new_left_cached_nonlin_attentions_encoder_6,
            new_left_cached_values_1_encoder_6,
            new_left_cached_values_2_encoder_6,
            new_left_cached_convolutions_1_encoder_6,
            new_left_cached_convolutions_2_encoder_6,

            new_processed_len,
        )


def get_init_states(
    input_dim: int,
    num_encoder_layers: list[int],
    downsample_left_pad_frames: list[int],
    encoder_dims: list[int],
    query_dims: list[int],
    value_dims: list[int],
    head_dims: list[int],
    convolution_left_pad_frames: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Get initial states for the Zipformer2 encoder. The method generates a list of torch tensors,
    where the first tensor corresponds to a subsampling module left cache. Next, for each
    Zipformer2Encoder module we add six cache tensors that are essential for multi-head attention
    and convolution modules. Finally, at the end we append a total processed frames tensor,
    initialized with zero.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    num_encoder_layers : list[int]
        The number of Zipformer2EncoderLayer modules for each Zipformer2Encoder stack.
    downsample_left_pad_frames : list[int]
        The multi-head attention left context cache frames after downsampling.
    encoder_dims : list[int]
        The embedding dimension for each Zipformer2Encoder stack.
    query_dims : list[int]
        The multi-head attention query dimension for each Zipformer2Encoder stack.
    value_dims : list[int]
        The multi-head attention value dimension for each Zipformer2Encoder stack.
    head_dims : list[int]
        The non-linear attention head dimension for each Zipformer2Encoder stack.
    convolution_left_pad_frames : list[int]
        The convolution modules left padding number of frames for each Zipformer2Encoder stack.
    device : torch.device
        The device used to store cache tensors. Should be
        either torch.device("cpu") or torch.device("cuda").

    Returns
    -------
    list[torch.Tensor[torch.float32 | torch.int32]]
        A list of left cache tensors.
        - A subsampling module left cache tensor of shape (1, 10, input_dim)
        - The first Zipformer2Encoder cached attention key tensor of the left context in each
          Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_1, 1, left_context_len_1, query_dim_1).
        - The first Zipformer2Encoder left context cached attention tensor for the non-linear
          attention module in each Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_1, 1, left_context_len_1, head_dim_1).
        - The first Zipformer2Encoder cached left context tensor for the first self-attention
          module in each Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_1, 1, left_context_len_1, value_dim_1).
        - The first Zipformer2Encoder cached left context tensor for the second self-attention
          module in each Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_1, 1, left_context_len_1, value_dim_1).
        - The first Zipformer2Encoder cached left context tensor for the first convolution module
          in each Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_1, 1, encoder_dim_1, conv_left_pad_1).
        - The first Zipformer2Encoder cached left context tensor for the second convolution module
          in each Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_1, 1, encoder_dim_1, conv_left_pad_1).
                                            .
                                            .
                                            .
        - The sixth Zipformer2Encoder cached left context tensor for the second convolution module
          in each Zipformer2EncoderLayer of the stack.
          The tensor is of shape (num_layers_6, 1, encoder_dim_6, conv_left_pad_6).
        - The processed length integer tensor initialized with a single zero element.
          The tensor is of shape (1,).
    """
    # pylint: disable=too-many-locals

    if not (
        len(num_encoder_layers)
        == len(downsample_left_pad_frames)
        == len(encoder_dims)
        == len(query_dims)
        == len(value_dims)
        == len(head_dims)
        == len(convolution_left_pad_frames)
    ):
        raise ValueError(
            'It is required that all encoder parameter lists have the same '
            'length, but got following parameter list lengths:\n'
            f'len(num_encoder_layers) == {len(num_encoder_layers)}\n'
            f'len(downsample_left_pad_frames) == {len(downsample_left_pad_frames)}\n'
            f'len(encoder_dims) == {len(encoder_dims)}\n'
            f'len(query_dims) == {len(query_dims)}\n'
            f'len(value_dims) == {len(value_dims)}\n'
            f'len(nonlin_attn_head_dims) == {len(head_dims)}\n'
            f'len(convolution_left_pad_frames) == {len(convolution_left_pad_frames)}.',
        )

    states = [subsampling_get_init_states(input_dim, device)]
    for i, num_layers in enumerate(num_encoder_layers):

        encoder_dim = encoder_dims[i]
        query_dim = query_dims[i]
        value_dim = value_dims[i]
        head_dim = head_dims[i]
        left_context_len = downsample_left_pad_frames[i]
        left_cache_len = convolution_left_pad_frames[i]

        # batch size is 1
        states += [
            torch.zeros(
                num_layers, 1, left_context_len, query_dim, dtype=torch.float32, device=device,
            ),
            torch.zeros(
                num_layers, 1, left_context_len, head_dim, dtype=torch.float32, device=device,
            ),
            torch.zeros(
                num_layers, 1, left_context_len, value_dim, dtype=torch.float32, device=device,
            ),
            torch.zeros(
                num_layers, 1, left_context_len, value_dim, dtype=torch.float32, device=device,
            ),
            torch.zeros(
                num_layers, 1, encoder_dim, left_cache_len, dtype=torch.float32, device=device,
            ),
            torch.zeros(
                num_layers, 1, encoder_dim, left_cache_len, dtype=torch.float32, device=device,
            ),
        ]

    states.append(torch.zeros(1, dtype=torch.int32, device=device))

    return states


class Zipformer2EncoderLayer(torch.nn.Module):
    """
    Zipformer2EncoderLayer module, the basic block of Zipformer2Encoder encoder stack.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        cnn_module_kernel: int,
        left_context_len: int,
        right_context_len: int,
        device: torch.device,
    ) -> None:
        """
        Zipformer2EncoderLayer initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same for input
            and output of this module.
        pos_dim : int
            The dimension of the relative positional embedding.
        num_heads : int
            The number of heads for attention weights and self-attention.
        query_head_dim : int
            The dimension of the query and key per attention head in attention weights.
        pos_head_dim: int
            The dimension of the projected positional encoding
            per attention head in attention weights.
        value_head_dim : int
            The dimension of the value per attention head in self-attention.
        feedforward_dim : int
            The hidden dimension of the feedforward modules.
        cnn_module_kernel : int
            The kernel size of the convolution modules.
        left_context_len : int
            The module left context number of subsampled frames.
        right_context_len : int
            The module right context number of subsampled frames.
            Used to update attention and convolution left caches.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """
        # pylint: disable=too-many-arguments

        super().__init__()

        self.left_context_len = left_context_len

        # self.bypass implements the whole layer skipping.
        self.bypass = BypassModule(embed_dim, device)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, device)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim, num_heads, query_head_dim, pos_head_dim, right_context_len, device,
        )

        self.self_attn1 = SelfAttention(
            embed_dim, num_heads, value_head_dim, right_context_len, device,
        )
        self.self_attn2 = SelfAttention(
            embed_dim, num_heads, value_head_dim, right_context_len, device,
        )

        self.nonlin_attention = NonlinAttention(
            embed_dim, 3 * embed_dim // 4, right_context_len, device,
        )

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, device)
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, device)
        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, device)

        self.conv_module1 = ConvolutionModule(
            embed_dim, cnn_module_kernel, right_context_len, device,
        )
        self.conv_module2 = ConvolutionModule(
            embed_dim, cnn_module_kernel, right_context_len, device,
        )

        self.norm = BiasNorm(embed_dim, device)

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        left_cached_key: torch.Tensor,
        left_cached_nonlin_attn: torch.Tensor,
        left_cached_val_1: torch.Tensor,
        left_cached_val_2: torch.Tensor,
        left_cached_conv_1: torch.Tensor,
        left_cached_conv_2: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Does a forward pass of the Zipformer2EncoderLayer module. Returns an output tensor with the
        same shape as input, and updated left caches for multiple attention and convolution
        mudules.

        Parameters
        ----------
        src : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, embed_dim). The module input.
        pos_emb : torch.Tensor[torch.float32]
            A positional embedding tensor
            of shape (1, left_context_len + 2 * seq_len - 1, pos_dim).
        left_cached_key : torch.Tensor[torch.float32]
            A cached attention key tensor of the left context
            of shape (1, left_context_len, query_dim).
        left_cached_nonlin_attn : torch.Tensor[torch.float32]
            A left context cached attention tensor for the non-linear attention module
            of shape (1, left_context_len, head_dim).
        left_cached_val_1 : torch.Tensor[torch.float32]
            A cached left context tensor for the first self-attention module
            of shape (1, left_context_len, value_dim).
        left_cached_val_2 : torch.Tensor[torch.float32]
            A cached left context for the second self-attention module
            of shape (1, left_context_len, value_dim).
        left_cached_conv_1 : torch.Tensor[torch.float32]
            A cached left context tensor for the first convolution module
            of shape (1, embed_dim, left_cache_len).
        left_cached_conv_2 : torch.Tensor[torch.float32]
            A cached left context tensor for the second convolution module
            of shape (1, embed_dim, left_cache_len).
        src_key_padding_mask : torch.Tensor[torch.bool]
            A boolean tensor of shape (1, seq_len_2). Positions that are True in this mask will be
            ignored as sources in the attention weighting and convolution modules.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
        ]
            A tuple of seven float tensors:
            - The module output of shape (1, seq_len, embed_dim).
              A tensor with the same shape as input.
            - The updated left context cached attention key tensor
              of shape (1, left_context_len, query_dim).
            - The updated left context cached attention tensor for the non-linear attention module
              of shape (1, left_context_len, head_dim).
            - The updated cached left context for the first self-attention module
              of shape (1, left_context_len, value_dim).
            - The updated cached left context for the second self-attention module
              of shape (1, left_context_len, value_dim).
            - The updated cached left context for the first convolution module
              of shape (1, embed_dim, left_cache_len).
            - The updated cached left context for the second convolution module
              of shape (1, embed_dim, left_cache_len).
        """

        src_orig = src

        # attn_weights: (1, num_heads, seq_len, seq_len_2)
        attn_weights, left_cached_key = self.self_attn_weights(
            src, pos_emb, left_cached_key, src_key_padding_mask,
        )
        src = src + self.feed_forward1(src)

        na, left_cached_nonlin_attn = self.nonlin_attention(
            src, attn_weights[:, 0], left_cached_nonlin_attn,
        )
        src = src + na

        self_attn, left_cached_val_1 = self.self_attn1(src, attn_weights, left_cached_val_1)
        src = src + self_attn

        src_conv, left_cached_conv_1 = self.conv_module1(
            src, left_cached_conv_1, src_key_padding_mask[:, self.left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn, left_cached_val_2 = self.self_attn2(src, attn_weights, left_cached_val_2)
        src = src + self_attn

        src_conv, left_cached_conv_2 = self.conv_module2(
            src, left_cached_conv_2, src_key_padding_mask[:, self.left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)

        src = self.norm(src)
        src = self.bypass(src_orig, src)

        return (
            src,
            left_cached_key,
            left_cached_nonlin_attn,
            left_cached_val_1,
            left_cached_val_2,
            left_cached_conv_1,
            left_cached_conv_2,
        )


class Zipformer2Encoder(torch.nn.Module):
    """
    Zipformer2Encoder is a stack of Zipformer2EncoderLayer modules.
    """

    def __init__(
        self,
        encoder_layer: torch.nn.Module,
        num_layers: int,
        embed_dim: int,
        pos_dim: int,
        pos_max_len: int,
        downsample: int,
        device: torch.device,
    ) -> None:
        """
        Zipformer2Encoder initialization.

        Parameters
        ----------
        encoder_layer : torch.nn.Module
            An instance of the Zipformer2EncoderLayer class.
        num_layers : int
            The number of encoder Zipformer2EncoderLayer modules in the stack.
        embed_dim : int
            The input and output embedding dimension. The embedding dimension is the same for
            input and output of this module.
        pos_dim : int
            The dimension for the relative positional embedding.
        downsample : int
            The downsampling factor of the module, the input will be downsampled in the beginning
            and upsampled back at the end.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.num_layers = num_layers
        self.downsample = SimpleDownsample(downsample, device)
        self.encoder_pos = CompactRelPositionalEncoding(
            pos_dim, pos_max_len, encoder_layer.left_context_len, device,
        )

        self.layers = torch.nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)],
        )
        self.upsample = SimpleUpsample(downsample)
        self.out_combiner = BypassModule(embed_dim, device)

    def forward(
        self,
        src: torch.Tensor,
        left_cached_keys: torch.Tensor,
        left_cached_nonlin_attentions: torch.Tensor,
        left_cached_values_1: torch.Tensor,
        left_cached_values_2: torch.Tensor,
        left_cached_convolutions_1: torch.Tensor,
        left_cached_convolutions_2: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Does a forward pass of the Zipformer2Encoder module. Returns an output tensor with the same
        shape as input, and updated left caches for multiple attention and convolution mudules.

        Parameters
        ----------
        src : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, embed_dim). The module input.
        left_cached_keys : torch.Tensor[torch.float32]
            A cached attention key tensor of the left context for each Zipformer2EncoderLayer.
            A tensor is of shape (num_layers, 1, left_context_len, query_dim).
        left_cached_nonlin_attentions : torch.Tensor[torch.float32]
            A left context cached attention tensor for the non-linear attention module of each
            Zipformer2EncoderLayer. A tensor is
            of shape (num_layers, 1, left_context_len, head_dim).
        left_cached_values_1 : torch.Tensor[torch.float32]
            A cached left context tensor for the first self-attention module of each
            Zipformer2EncoderLayer. A tensor is
            of shape (num_layers, 1, left_context_len, value_dim).
        left_cached_values_2 : torch.Tensor[torch.float32]
            A cached left context tensor for the second self-attention module of each
            Zipformer2EncoderLayer. A tensor is
            of shape (num_layers, 1, left_context_len, value_dim).
        left_cached_convolutions_1 : torch.Tensor[torch.float32]
            A cached left context tensor for the first convolution module of each
            Zipformer2EncoderLayer. A tensor is
            of shape (num_layers, 1, embed_dim, left_cache_len).
        left_cached_convolutions_2 : torch.Tensor[torch.float32]
            A cached left context tensor for the second convolution module of each
            Zipformer2EncoderLayer. A tensor is
            of shape (num_layers, 1, embed_dim, left_cache_len).
        src_key_padding_mask : torch.Tensor[torch.bool]
            A boolean tensor of shape (1, seq_len_2). Positions that are True in this mask will be
            ignored as sources in the attention weighting and convolution modules.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
        ]
            A tuple of seven float tensors:
            - The module output of shape (1, seq_len, embed_dim).
              A tensor with the same shape as input.
            - The updated cached attention key tensor of the left context for each
              Zipformer2EncoderLayer. A tensor is
              of shape (num_layers, 1, left_context_len, query_dim).
            - The updated left context cached attention tensor for the non-linear attention module
              of each Zipformer2EncoderLayer. A tensor is
              of shape (num_layers, 1, left_context_len, head_dim).
            - The updated cached left context tensor for the first self-attention module of each
              Zipformer2EncoderLayer. A tensor is
              of shape (num_layers, 1, left_context_len, value_dim).
            - The updated cached left context tensor for the second self-attention module of each
              Zipformer2EncoderLayer. A tensor is
              of shape (num_layers, 1, left_context_len, value_dim).
            - The updated cached left context tensor for the first convolution module of each
              Zipformer2EncoderLayer. A tensor is
              of shape (num_layers, 1, embed_dim, left_cache_len).
            - The updated cached left context tensor for the second convolution module of each
              Zipformer2EncoderLayer. A tensor is
              of shape (num_layers, 1, embed_dim, left_cache_len).
        """
        # pylint: disable=too-many-locals

        src_orig = src
        src = self.downsample(src)
        pos_emb = self.encoder_pos(src)

        new_left_cached_keys = torch.empty(
            left_cached_keys.shape, dtype=torch.float32, device=left_cached_keys.device,
        )
        new_left_cached_nonlin_attentions = torch.empty(
            left_cached_nonlin_attentions.shape,
            dtype=torch.float32,
            device=left_cached_nonlin_attentions.device,
        )
        new_left_cached_values_1 = torch.empty(
            left_cached_values_1.shape, dtype=torch.float32, device=left_cached_values_1.device,
        )
        new_left_cached_values_2 = torch.empty(
            left_cached_values_2.shape, dtype=torch.float32, device=left_cached_values_2.device,
        )
        new_left_cached_convolutions_1 = torch.empty(
            left_cached_convolutions_1.shape,
            dtype=torch.float32,
            device=left_cached_convolutions_1.device,
        )
        new_left_cached_convolutions_2 = torch.empty(
            left_cached_convolutions_2.shape,
            dtype=torch.float32,
            device=left_cached_convolutions_2.device,
        )

        for i, mod in enumerate(self.layers):
            (
                src,
                new_left_cached_keys[i],
                new_left_cached_nonlin_attentions[i],
                new_left_cached_values_1[i],
                new_left_cached_values_2[i],
                new_left_cached_convolutions_1[i],
                new_left_cached_convolutions_2[i],
            ) = mod(
                src,
                pos_emb,
                left_cached_keys[i],
                left_cached_nonlin_attentions[i],
                left_cached_values_1[i],
                left_cached_values_2[i],
                left_cached_convolutions_1[i],
                left_cached_convolutions_2[i],
                src_key_padding_mask,
            )

        src = self.upsample(src)

        # Remove any extra frames that are not a multiple of downsample_factor
        src = src[:, : src_orig.size(1)]
        src = self.out_combiner(src_orig, src)

        return (
            src,
            new_left_cached_keys,
            new_left_cached_nonlin_attentions,
            new_left_cached_values_1,
            new_left_cached_values_2,
            new_left_cached_convolutions_1,
            new_left_cached_convolutions_2,
        )


class BypassModule(torch.nn.Module):
    """
    A bypass module that implements a learnable bypass scale for each input channel.
    """

    def __init__(self, num_channels: int, device: torch.device) -> None:
        """
        BypassModule initialization.

        Parameters
        ----------
        num_channels : int
            The number of input channels, corresponds to the number of learnable bypass scales.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()
        self.bypass_scale = torch.nn.Parameter(
            torch.ones(num_channels, dtype=torch.float32, device=device),
        )

    def forward(self, x_early: torch.Tensor, x_later: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the BypassModule module.

        Parameters
        ----------
        x_early : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, num_channels).
            The module input that will be propagated with (1 - self.bypass_scale) weight.
        x_later : torch.Tensor[torch.float32]
            An input float tensor of shape (1, seq_len, num_channels).
            The module input that will be propagated with self.bypass_scale weight.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape (1, seq_len, num_channels). The shape is the same for x_early
            and x_later. The output of the module is x_early bypassed and added to x_later.
        """

        # It's just a slightly more efficient implementation of
        # (1.0 - self.bypass_scale) * x_early + self.bypass_scale * x_later
        return x_early + (x_later - x_early) * self.bypass_scale


class SimpleDownsample(torch.nn.Module):
    """
    A downsample layer, does downsampling by weighted sum aggregation.
    """

    def __init__(self, downsample: int, device: torch.device) -> None:
        """
        SimpleDownsample initialization.

        Parameters
        ----------
        downsample : int
            The module downsampling factor.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.zeros(downsample, 1, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the SimpleDownsample module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, num_channels).
            The module input that will be downsampled.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape
            (1, (seq_len + downsample - 1) // downsample, num_channels).
            The downsampled output of the module.
        """

        downsample = self.weights.size(0)
        if downsample == 1:
            return x

        batch_size, seq_len, in_channels = x.size()  # batch_size is 1
        downsampled_seq_len = (seq_len + downsample - 1) // downsample

        # Pad to an exact multiple of downsample. Right-pad x, repeating the last element.
        pad_frames = downsampled_seq_len * downsample - seq_len
        if pad_frames > 0:
            pad = x[:, seq_len - 1:, :].expand(batch_size, pad_frames, in_channels)
            x = torch.cat((x, pad), dim=1)

        # (1, seq_len, in_channels) -> (1, seq_len // downsample, downsample, in_channels)
        x = x.reshape(batch_size, downsampled_seq_len, downsample, in_channels)

        x = torch.sum(x * self.weights, dim=2)

        return x


class SimpleUpsample(torch.nn.Module):
    """
    An upsample layer, does upsampling by repeating the input frames.
    """

    def __init__(self, upsample: int) -> None:
        """
        SimpleUpsample initialization.

        Parameters
        ----------
        upsample : int
            The module upsampling factor.
        """

        super().__init__()
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the SimpleUpsample module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, num_channels).
            The module input that will be upsampled.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape (1, seq_len * upsample, num_channels).
            The upsampled output of the module.
        """

        if self.upsample == 1:
            return x

        x = torch.repeat_interleave(x, self.upsample, dim=1)

        return x


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module. This version is "compact" meaning it is able to encode the
    important information about the relative positions in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets
    (e.g. 1000 vs. 1001) make very little difference to the embedding. Such differences were
    potentially important when encoding absolute position, but not important when encoding relative
    position because there is now no need to compare two large offsets with each other.

    This implementation works by projecting the interval [-infinity, infinity] to a finite interval
    using the torch.atan() function before doing the fourier transform of that fixed interval.
    The torch.atan() function would compress the "long tails" too small, making it hard to
    distinguish between different magnitudes of large offsets. To mitigate this a logarithmic
    function is used to compress large offsets to a smaller range before applying torch.atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets
    as long as they are quite close to the origin, e.g. abs(offset) <= sqrt(embedding_dim).
    """

    def __init__(
        self, embed_dim: int, max_length: int, left_context_len: int, device: torch.device,
    ) -> None:
        """
        CompactRelPositionalEncoding initialization.

        Parameters
        ----------
        embed_dim : int
            The positional embedding dimension.
        max_length : int
            The maximum length of the input that this module will be able to handle after
            initialization without positional embeddings expansion. In case of longer input the
            positional embeddings will be re-computed to adjust bigger length.
        left_context_len : int
            Length of cached left context.
        device : torch.device
            The device used to store the layer positional embeddings.
            Should be either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError(
                'Embedding dimension for CompactRelPositionalEncoding '
                f'should be an even number, but got {embed_dim}.',
            )

        self.embed_dim = embed_dim
        self.left_context_len = left_context_len
        self.pos_emb = self.create_pos_emb(max_length, device)

    def create_pos_emb(self, max_length: int, device: torch.device) -> torch.Tensor:
        """
        Creates a relative positional embeddings based on the maximum length.
        This method is used to create positional embeddings with a
        sufficiently long temporal axes during module initialization.
        We want it to be big enough to avoid getting input x that is longer
        than self.pos_emb during inference. On the other hand, we want
        to initialize it with the smallest maximum length possible to consume
        less memory.

        Parameters
        ----------
        max_length : int
            The maximum length of the input that can be handeled by this layer. Increasing this
            will let to process bigger input (speaking of temporal dimension), but will also
            increase the memory consumption.
        device : torch.device
            The device used to store the positional embeddings.
            Should be either torch.device("cpu") or torch.device("cuda").

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape (2 * max_length - 1, embed_dim).
            Relative positional embeddings.
        """

        # if max_length == 4, the x would contain [-3, -2, -1, 0, 1, 2, 3]
        x = torch.arange(-max_length + 1, max_length, dtype=torch.float32, device=device)

        # Compression length is an arbitrary heuristic, if it is larger we have more resolution for
        # small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5

        # Compressing x within the next line of code, similarly to uncompressed x, it goes from
        # -infinity to infinity as the sequence length goes from -infinity to infinity, but it does
        # so more slowly than sequence length for the large absolute values of sequence length.
        # The formula is chosen so that d(x_compressed) / dx is equal to 1 around x == 0,
        # which is important.
        x = compression_length * torch.sign(x) * torch.log(torch.abs(x) / compression_length + 1.0)

        # results between -pi and pi
        x = torch.atan(2.0 * torch.pi * x / self.embed_dim)

        freqs = torch.arange(1, self.embed_dim // 2 + 1, dtype=torch.float32, device=device)
        x = x.unsqueeze(1) * freqs

        pos_emb = torch.zeros(x.size(0), self.embed_dim, dtype=torch.float32, device=device)
        pos_emb[:, 0::2] = torch.cos(x)
        pos_emb[:, 1::2] = torch.sin(x)
        pos_emb[:, self.embed_dim - 1] = 1.0  # for bias.

        return pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the CompactRelPositionalEncoding module.
        Returns a relative positional embeddings based on the input x temporal dimension.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            An input float tensor of shape (1, seq_len, embed_dim).
            The module input. It's shape will be used to construct relative positional embeddings.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape (1, self.left_context_len + 2 * seq_len - 1, embed_dim).
            Relative positional embeddings.
        """

        if self.pos_emb.size(0) < 2 * (x.size(1) + self.left_context_len) - 1:
            self.pos_emb = self.create_pos_emb(x.size(1) + self.left_context_len, x.device)

        # Length of negative side: x.size(1) + self.left_context_len.
        # Length of positive side: x.size(1).
        pos_emb = self.pos_emb[
            self.pos_emb.size(0) // 2 - x.size(1) - self.left_context_len + 1:
            self.pos_emb.size(0) // 2 + x.size(1)
        ].unsqueeze(0).repeat(x.size(0), 1, 1)

        # (1, left_context_len + 2 * seq_len - 1, embed_dim),
        # i. e. (batch_size, pos_len, embed_dim).
        return pos_emb


class RelPositionMultiheadAttentionWeights(torch.nn.Module):
    """
    Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example,
    the SelfAttention module which allows you to compute conventional self-attention.

    This is a quite heavily modified from:
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context".
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        right_context: int,
        device: torch.device,
    ) -> None:
        """
        RelPositionMultiheadAttentionWeights initialization.

        Parameters
        ----------
        embed_dim : int
            The embedding dimension. The number of channels at the input to this module.
        pos_dim : int
            A dimension of the positional embeddings.
        num_heads : int
            The number of attention heads to compute weights.
        query_head_dim : int
            The dimension of the query and key per head.
        pos_head_dim : int
            The dimension of the projected positional encoding per head.
        right_context : int
            The module look ahead future context, used to update left
            cached attention key correctly.
        device : torch.device
            The device used to store the layer positional embeddings. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.right_context = right_context

        in_proj_dim = (2 * query_head_dim + pos_head_dim) * num_heads
        self.in_proj = torch.nn.Linear(embed_dim, in_proj_dim, device=device)

        # Linear transformation for positional encoding.
        self.linear_pos = torch.nn.Linear(
            pos_dim, num_heads * pos_head_dim, bias=False, device=device,
        )

    def forward(
        self, x: torch.Tensor,
        pos_emb: torch.Tensor,
        left_cached_key: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the RelPositionMultiheadAttentionWeights module.
        Returns attention weights and updated cached attention key tensor of the left context.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, embed_dim). The module input.
        pos_emb : torch.Tensor[torch.float32]
            A positional embedding tensor
            of shape (1, left_context_len + 2 * seq_len - 1, pos_dim).
        left_cached_key : torch.Tensor[torch.float32]
            A cached attention key tensor of the left context
            of shape (1, left_context_len, key_dim).
        key_padding_mask : torch.Tensor[torch.bool]
            A boolean tensor of shape (1, seq_len_2). Positions that are True in this mask will be
            ignored as sources in the attention weighting.

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
            A tuple of two float tensors:
            - attention weights, of shape (1, hum_heads, seq_len, seq_len_2)
              interpreted as (1, hum_heads, tgt_seq_len, src_seq_len).
            - updated cached attention key tensor of the left context
              of shape (1, left_context_len, key_dim).
        """
        # pylint: disable=too-many-locals

        batch_size = x.size(0)  # batch size is 1
        seq_len = x.size(1)
        x = self.in_proj(x)

        query_dim = self.query_head_dim * self.num_heads

        # Self-attention.
        q = x[:, :, :query_dim]
        k = x[:, :, query_dim: 2 * query_dim]
        # p is the position-encoding query.
        p = x[:, :, 2 * query_dim:]

        # Pad key with cached left context.
        k = torch.cat((left_cached_key, k), dim=1)
        # Update cached left contexts
        seq_len_2 = k.size(1)  # left_context_len + seq_len
        left_cached_key = k[
            :,
            seq_len_2 - self.right_context - left_cached_key.size(1):
            seq_len_2 - self.right_context,
        ]

        q = q.reshape(batch_size, seq_len, self.num_heads, self.query_head_dim)
        p = p.reshape(batch_size, seq_len, self.num_heads, self.pos_head_dim)
        k = k.reshape(batch_size, seq_len_2, self.num_heads, self.query_head_dim)

        # seq_len refers to target, seq_len_2 refers to source.
        q = q.permute(0, 2, 1, 3)  # (1, hum_heads, seq_len, query_head_dim)
        p = p.permute(0, 2, 1, 3)  # (1, hum_heads, seq_len, pos_head_dim)
        k = k.permute(0, 2, 3, 1)  # (1, hum_heads, key_head_dim, seq_len_2)

        attn_scores = torch.matmul(q, k)  # (1, hum_heads, seq_len, seq_len_2)

        pos_len = pos_emb.size(1)  # left_context_len + 2 * seq_len - 1
        # (1, pos_len, num_heads * pos_head_dim)
        pos_emb = self.linear_pos(pos_emb)
        pos_emb = pos_emb.reshape(
            batch_size, pos_len, self.num_heads, self.pos_head_dim,
        ).permute(0, 2, 3, 1)  # (1, hum_heads, pos_head_dim, pos_len)

        # (1, hum_heads, seq_len, pos_head_dim) x (1, hum_heads, pos_head_dim, pos_len) ->
        # -> (1, hum_heads, seq_len, pos_len) where pos_len represents relative position.
        pos_scores = torch.matmul(p, pos_emb)

        # Now we need to perform the relative shift of the pos_scores, to do that we need to add
        # a column of zeros to the left side of the last dimension and perform the relative shift.
        pos_scores_pad = torch.zeros(
            pos_scores.size(0), pos_scores.size(1), pos_scores.size(2), 1,
            dtype=torch.float32,
            device=pos_scores.device,
        )
        # (1, hum_heads, seq_len, pos_len + 1)
        pos_scores = torch.cat((pos_scores_pad, pos_scores), dim=3)
        pos_scores = pos_scores.reshape(
            batch_size, self.num_heads, pos_len + 1, seq_len,
        )  # (1, hum_heads, pos_len + 1, seq_len)
        # Now drop the extra row that had been added over padding and reshape.
        pos_scores = pos_scores[:, :, 1:].reshape(
            batch_size, self.num_heads, seq_len, pos_len,
        )  # (1, hum_heads, seq_len, pos_len)

        # (1, hum_heads, seq_len, seq_len_2)
        attn_scores = attn_scores + pos_scores[:, :, :, : attn_scores.size(3)]

        # (1, seq_len_2) -> (1, 1, 1, seq_len_2) to make it broadcastable to attn_scores shape.
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        attn_scores = torch.masked_fill(attn_scores, key_padding_mask, -1000.0)
        attn_weights = torch.softmax(attn_scores, dim=3)

        return attn_weights, left_cached_key


class SelfAttention(torch.nn.Module):
    """
    The simplest possible attention module. This one works with pre-computed attention weights,
    e.g. as computed by RelPositionMultiheadAttentionWeights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
        right_context: int,
        device: torch.device,
    ) -> None:
        """
        SelfAttention initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same for input
            and output of this module.
        num_heads : int
            The number of attention heads.
        value_head_dim : int
            The dimension of the value per head.
        right_context : int
            The module look ahead future context, used to update left cached
            attention value correctly.
        device : torch.device
            The device used to store the layer positional embeddings.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, num_heads * value_head_dim, device=device)
        self.out_proj = torch.nn.Linear(num_heads * value_head_dim, embed_dim, device=device)
        self.right_context = right_context

    def forward(
        self, x: torch.Tensor, attn_weights: torch.Tensor, left_cached_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the SelfAttention module. Returns attention weighted input tensor
        and updated cached attention value tensor of the left context.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, embed_dim). The module input.
        attn_weights : torch.Tensor[torch.float32]
            The tensor of shape (1, num_heads, seq_len, seq_len_2), with (seq_len, seq_len_2)
            being interpreted as (tgt_seq_len, src_seq_len). Expect attn_weights.sum(dim=3) == 1.0.
        left_cached_val : torch.Tensor[torch.float32]
            The cached attention value tensor of the left context
            of shape (1, left_context_len, value_dim).

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
            A tuple of two float tensors:
            - attention weighted output of shape (1, seq_len, embed_dim).
              A tensor with the same shape as input x.
            - updated cached attention value tensor of the left context
              of shape (1, left_context_len, value_dim).
        """

        batch_size = x.size(0)  # batch size is 1
        num_heads = attn_weights.size(1)

        x = self.in_proj(x)  # (1, seq_len, num_heads * value_head_dim)

        x = torch.cat((left_cached_val, x), dim=1)
        # Update cached left contexts
        left_cached_val = x[
            :,
            x.size(1) - self.right_context - left_cached_val.size(1):
            x.size(1) - self.right_context,
        ]

        x = x.reshape(batch_size, x.size(1), num_heads, x.size(2) // num_heads).permute(0, 2, 1, 3)

        # (1, num_heads, seq_len, seq_len_2) x (1, num_heads, seq_len_2, value_head_dim) ->
        # -> (1, num_heads, seq_len, value_head_dim)
        x = torch.matmul(attn_weights, x)

        # (1, num_heads, seq_len, value_head_dim) -> (1, seq_len, num_heads, value_head_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, x.size(1), num_heads * x.size(3))

        # returned value is of shape (1, seq_len, embed_dim), like the input.
        x = self.out_proj(x)

        return x, left_cached_val


class FeedforwardModule(torch.nn.Module):
    """
    Feedforward module in Zipformer2 encoder.
    """

    def __init__(self, embed_dim: int, feedforward_dim: int, device: torch.device) -> None:
        """
        FeedforwardModule initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same for input
            and output of this module.
        feedforward_dim : int
            The module hidden dimension.
        device : torch.device
            The device used to store the layer weights. should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, feedforward_dim, device=device)
        self.activation = SwooshL()
        self.out_proj = torch.nn.Linear(feedforward_dim, embed_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the FeedforwardModule module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            A float tensor of shape (1, seq_len, embed_dim). The module input.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape (1, seq_len, embed_dim).
            The module output has the same shape as input.
        """

        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)

        return x


class NonlinAttention(torch.nn.Module):
    """
    This is like the ConvolutionModule, but refactored so that we use multiplication by attention
    weights (borrowed from the RelPositionMultiheadAttentionWeights module) instead of actual
    convolution. We also took out the second nonlinearity, the one after the attention mechanism.
    """

    def __init__(
        self, embed_dim: int, att_dim: int, right_context: int, device: torch.device,
    ) -> None:
        """
        NonlinAttention initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same for input
            and output of this module.
        att_dim : int
            The attention output dimension of this module.
        right_context : int
            The module look ahead future context, used to update left cache
            correctly.
        device : torch.device
            The device used to store the positional embeddings.
            Should be either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, att_dim * 3, device=device)
        self.out_proj = torch.nn.Linear(att_dim, embed_dim, device=device)
        self.right_context = right_context

    def forward(
        self, x: torch.Tensor, attn_weights: torch.Tensor, left_cached_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the NonlinAttention module. Returns attention weighted input tensor
        and updated attention input tensor cache of the left context.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            An input float tensor of shape (1, seq_len, embed_dim).
        attn_weights : torch.Tensor[torch.float32]
            A tensor of shape (1, seq_len, seq_len_2), that corresponds to a single attention head
            with (seq_len, seq_len_2) being interpreted as (tgt_seq_len, src_seq_len).
            Expected attn_weights.sum(dim=2) == 1.0.
            Note: the first dimension here corresponds to a batch size.
        left_cached_x : torch.Tensor[torch.float32]
            A cached attention tensor of the left context of shape (1, left_context_len, att_dim).

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
            A tuple of two float tensors:
            - attention weighted output of shape (1, seq_len, embed_dim).
              A tensor with the same shape as input x.
            - updated cached attention tensor of the left context
              of shape (1, left_context_len, att_dim).
        """

        x = self.in_proj(x)

        s, x, y = x.chunk(3, dim=2)

        x = x * torch.tanh(s)

        x = torch.cat((left_cached_x, x), dim=1)
        # Update cached tensor
        left_cached_x = x[
            :,
            x.size(1) - self.right_context - left_cached_x.size(1):
            x.size(1) - self.right_context,
        ]

        # (1, seq_len, seq_len_2) x (1, seq_len_2, att_dim) -> (1, seq_len, att_dim)
        x = torch.matmul(attn_weights, x)
        x = x * y

        x = self.out_proj(x)

        return x, left_cached_x


class ConvolutionModule(torch.nn.Module):
    """
    ConvolutionModule in Zipformer2 encoder.
    """

    def __init__(
        self, embed_dim: int, kernel_size: int, right_context: int, device: torch.device,
    ) -> None:
        """
        ConvolutionModule initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension, also the number of channels of convolution
            modules. The embedding dmension is the same for input and output of this module.
        kernel_size : int
            The kernel size of the depthwise convolution module.
        right_context : int
            The module look ahead future context, used to update
            causal depthwise convolution left cache correctly.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                'ConvolutionModule kernerl size should be '
                f'an odd number but got {kernel_size} instead.',
            )

        self.in_proj = torch.nn.Linear(embed_dim, 2 * embed_dim, device=device)
        self.depthwise_conv = ChunkCausalDepthwiseConv1d(
            embed_dim, kernel_size, right_context, device,
        )

        self.activation = SwooshR()
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, device=device)

    def forward(
        self, x: torch.Tensor, left_cache: torch.Tensor, src_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the ConvolutionModule module. Returns processed tensor of the same
        shape as input and updated cached convolution tensor of the left context.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float tensor of shape (1, seq_len, embed_dim). The module input.
        left_cache : torch.Tensor[torch.float32]
            A cached convolution tensor of the left context
            of shape (1, embed_dim, left_cache_len).
        src_key_padding_mask : torch.Tensor[torch.bool]
            The mask for the source keys of shape (1, seq_len),
            contains True in masked positions that will be ignored.

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
            A tuple of two float tensors:
            - module output of shape (1, seq_len, embed_dim).
              A tensor with the same shape as input x.
            - updated cached convolution tensor of the left context
              of shape (1, embed_dim, left_cache_len).
        """

        x = self.in_proj(x)  # (1, seq_len, 2 * embed_dim)

        x, s = x.chunk(2, dim=2)
        x = x * torch.sigmoid(s)  # (1, seq_len, embed_dim)

        x = torch.masked_fill(x, src_key_padding_mask.unsqueeze(2), 0.0)

        # exchange the temporal dimension and the feature dimension for depthwise convolution.
        x = x.permute(0, 2, 1)  # (1, embed_dim, seq_len).
        x, left_cache = self.depthwise_conv(x, left_cache)
        x = x.permute(0, 2, 1)  # (1, seq_len, embed_dim)

        x = self.activation(x)
        x = self.out_proj(x)  # (1, seq_len, embed_dim)

        return x, left_cache


def _test_zipformer_main(causal: bool = False):
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.

    c = Zipformer2(
        encoder_dim=(64, 96),
        encoder_unmasked_dim=(48, 64),
        num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
