# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from torchaudio.models import Emformer as _Emformer
from torchaudio.models.rnnt import _TimeReduction

LOG_EPSILON = math.log(1e-10)


class Emformer(EncoderInterface):
    """This is just a simple wrapper around torchaudio.models.Emformer.
    We may replace it with our own implementation some time later.
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        segment_length: int,
        left_context_length: int,
        right_context_length: int,
        max_memory_size: int = 0,
        dropout: float = 0.1,
        subsampling_factor: int = 4,
        vgg_frontend: bool = False,
    ) -> None:
        """
        Args:
          num_features:
            The input dimension of the model.
          output_dim:
            The output dimension of the model.
          d_model:
            Attention dimension.
          nhead:
            Number of heads in multi-head attention.
          dim_feedforward:
            The output dimension of the feedforward layers in encoder.
          num_encoder_layers:
            Number of encoder layers.
          segment_length:
            Number of frames per segment.
          left_context_length:
            Number of frames in the left context.
          right_context_length:
            Number of frames in the right context.
          max_memory_size:
            TODO.
          dropout:
            Dropout in encoder.
          subsampling_factor:
            Number of output frames is num_in_frames // subsampling_factor.
          vgg_frontend:
            True to use vgg style frontend for subsampling.
        """
        super().__init__()

        self.subsampling_factor = subsampling_factor

        self.time_reduction = _TimeReduction(stride=subsampling_factor)
        self.in_linear = nn.Linear(num_features * subsampling_factor, d_model)

        self.right_context_length = right_context_length

        assert right_context_length % subsampling_factor == 0
        assert segment_length % subsampling_factor == 0
        assert left_context_length % subsampling_factor == 0

        left_context_length = left_context_length // subsampling_factor
        right_context_length = right_context_length // subsampling_factor
        segment_length = segment_length // subsampling_factor

        self.model = _Emformer(
            input_dim=d_model,
            num_heads=nhead,
            ffn_dim=dim_feedforward,
            num_layers=num_encoder_layers,
            segment_length=segment_length,
            dropout=dropout,
            activation="relu",
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=max_memory_size,
            weight_init_scale_strategy="depthwise",
            tanh_on_mem=False,
            negative_inf=-1e8,
        )

        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(d_model, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            Input features of shape (N, T, C).
          x_lens:
            A int32 tensor of shape (N,) containing valid frames in `x` before
            padding. We have `x.size(1) == x_lens.max()`
        Returns:
          Return a tuple containing two tensors:

            - encoder_out, a tensor of shape (N, T', C)
            - encoder_out_lens, a int32 tensor of shape (N,) containing the
              valid frames in `encoder_out` before padding
        """
        x = nn.functional.pad(
            x,
            # (left, right, top, bottom)
            # left/right are for the channel dimension, i.e., axis 2
            # top/bottom are for the time dimension, i.e., axis 1
            (0, 0, 0, self.right_context_length),
            value=LOG_EPSILON,
        )  # (N, T, C) -> (N, T+right_context_length, C)

        x, x_lens = self.time_reduction(x, x_lens)
        x = self.in_linear(x)

        emformer_out, emformer_out_lens = self.model(x, x_lens)
        logits = self.encoder_output_layer(emformer_out)

        return logits, emformer_out_lens

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
    ):
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 2-D tensor of shap containing the number of valid frames for each
            element in `x` before padding.
          states:
            Internal states of the model.
        Returns:
          Return a tuple containing 3 tensors:
            - encoder_out, a 3-D tensor of shape (N, T, C)
            - encoder_out_lens: a 1-D tensor of shape (N,)
            - next_state, internal model states for the next chunk
        """
        x, x_lens = self.time_reduction(x, x_lens)
        x = self.in_linear(x)

        emformer_out, emformer_out_lens, states = self.model.infer(
            x, x_lens, states
        )

        logits = self.encoder_output_layer(emformer_out)

        return logits, emformer_out_lens, states
