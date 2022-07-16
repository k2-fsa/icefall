# Copyright    2022  Xiaomi Corp.        (authors: Zengwei Yao)
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
import math
import warnings
from typing import List, Optional, Tuple

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
    ScaledLSTM,
)
from torch import Tensor, nn


class RNNEncoderLayer(nn.Module):
    """
    RNNEncoderLayer is made up of lstm and feedforward networks.

    Args:
      d_model:
        The number of expected features in the input (required).
      dim_feedforward:
        The dimension of feedforward network model (default=2048).
      dropout:
        The dropout value (default=0.1).
      layer_dropout:
        The dropout value for model-level warmup (default=0.075).
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ) -> None:
        super(RNNEncoderLayer, self).__init__()
        self.layer_dropout = layer_dropout
        self.d_model = d_model

        self.lstm = ScaledLSTM(
            input_size=d_model, hidden_size=d_model, dropout=0.0
        )
        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )
        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).  # noqa
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, warmup: float = 1.0) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          warmup:
            It controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.
        """
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

        # lstm module
        src_lstm = self.lstm(src)[0]
        src = src + self.dropout(src_lstm)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src

    @torch.jit.export
    def infer(
        self, src: Tensor, states: Tuple[Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        Pass the input through the encoder layer.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          states:
            Its shape is (2, 1, N, E).
            states[0] and states[1] are cached hidden state and cell state,
            respectively.
        """
        assert not self.training
        assert states.shape == (2, 1, src.size(1), src.size(2))

        # lstm module
        # The required shapes of h_0 and c_0 are both (1, N, E)
        src_lstm, new_states = self.lstm(src, states.unbind(dim=0))
        new_states = torch.stack(states, dim=0)
        src = src + self.dropout(src_lstm)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        return src, new_states


class RNNEncoder(nn.Module):
    """
    RNNEncoder is a stack of N encoder layers.

    Args:
      encoder_layer:
        An instance of the RNNEncoderLayer() class (required).
      num_layers:
        The number of sub-encoder-layers in the encoder (required).
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super(RNNEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src: Tensor, warmup: float = 1.0) -> Tensor:
        """
        Pass the input through the encoder layer in turn.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          warmup:
            It controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.
        """
        output = src

        for layer_index, mod in enumerate(self.layers):
            output = mod(output, warmup=warmup)

        return output

    @torch.jit.export
    def infer(
        self, src: Tensor, states: Tuple[Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        Pass the input through the encoder layer.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          states:
            Its shape is (2, num_layers, N, E).
            states[0] and states[1] are cached hidden states and cell states for
            all layers, respectively.
        """
        assert not self.training
        assert states.shape == (2, self.num_layers, src.size(1), src.size(2))

        new_states_list = []
        output = src
        for layer_index, mod in enumerate(self.layers):
            # new_states: (2, 1, N, E)
            output, new_states = mod.infer(
                output, states[:, layer_index : layer_index + 1, :, :]
            )
            new_states_list.append(new_states)

        return output, torch.cat(new_states_list, dim=1)
