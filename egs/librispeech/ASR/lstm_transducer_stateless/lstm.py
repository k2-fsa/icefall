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
from typing import Tuple

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv2d,
    ScaledLinear,
    ScaledLSTM,
)
from torch import nn


class RNN(EncoderInterface):
    """
    Args:
      num_features (int):
        Number of input features.
      subsampling_factor (int):
        Subsampling factor of encoder (convolution layers before lstm layers).
      d_model (int):
        Hidden dimension for lstm layers, also output dimension (default=512).
      dim_feedforward (int):
        Feedforward dimension (default=2048).
      num_encoder_layers (int):
        Number of encoder layers (default=12).
      dropout (float):
        Dropout rate (default=0.1).
      layer_dropout (float):
        Dropout value for model-level warmup (default=0.075).
    """

    def __init__(
        self,
        num_features: int,
        subsampling_factor: int,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ) -> None:
        super(RNN, self).__init__()

        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder_layers = num_encoder_layers
        self.d_model = d_model

        encoder_layer = RNNEncoderLayer(
            d_model, dim_feedforward, dropout, layer_dropout
        )
        self.encoder = RNNEncoder(encoder_layer, num_encoder_layers)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, T, C), where N is the batch size,
            T is the sequence length, C is the feature dimension.
          x_lens:
            A tensor of shape (N,), containing the number of frames in `x`
            before padding.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          A tuple of 2 tensors:
            - embeddings: its shape is (N, T', d_model), where T' is the output
              sequence lengths.
            - lengths: a tensor of shape (batch_size,) containing the number of
              frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1
        assert x.size(0) == lengths.max().item()

        x = self.encoder(x, warmup)

        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        return x, lengths

    @torch.jit.export
    def get_init_state(self, device: torch.device) -> torch.Tensor:
        """Get model initial state."""
        init_states = torch.zeros(
            (2, self.num_encoder_layers, self.d_model), device=device
        )
        return init_states

    @torch.jit.export
    def infer(
        self, x: torch.Tensor, x_lens: torch.Tensor, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, T, C), where N is the batch size,
            T is the sequence length, C is the feature dimension.
          x_lens:
            A tensor of shape (N,), containing the number of frames in `x`
            before padding.
          states:
            Its shape is (2, num_encoder_layers, N, E).
            states[0] and states[1] are cached hidden states and cell states for
            all layers, respectively.

        Returns:
          A tuple of 3 tensors:
            - embeddings: its shape is (N, T', d_model), where T' is the output
              sequence lengths.
            - lengths: a tensor of shape (batch_size,) containing the number of
              frames in `embeddings` before padding.
            - updated states, with shape of (2, num_encoder_layers, N, E).
        """
        assert not self.training
        assert states.shape == (
            2,
            self.num_encoder_layers,
            x.size(0),
            self.d_model,
        ), states.shape

        # lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1
        # we will cut off 1 frame on each side of encoder_embed output
        lengths -= 2

        embed = self.encoder_embed(x)
        embed = embed[:, 1:-1, :]
        embed = embed.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        x, states = self.encoder.infer(embed, states)

        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        return x, lengths, states


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

    def forward(self, src: torch.Tensor, warmup: float = 1.0) -> torch.Tensor:
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
        self, src: torch.Tensor, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # The required shapes of h_0 and c_0 are both (1, N, E).
        src_lstm, new_states = self.lstm(src, states.unbind(dim=0))
        new_states = torch.stack(new_states, dim=0)
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

    def forward(self, src: torch.Tensor, warmup: float = 1.0) -> torch.Tensor:
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
        self, src: torch.Tensor, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
