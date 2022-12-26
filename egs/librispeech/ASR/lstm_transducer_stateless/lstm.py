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
from typing import List, Optional, Tuple

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

LOG_EPSILON = math.log(1e-10)


def unstack_states(
    states: Tuple[torch.Tensor, torch.Tensor]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Unstack the lstm states corresponding to a batch of utterances into a list
    of states, where the i-th entry is the state from the i-th utterance.

    Args:
      states:
        A tuple of 2 elements.
        ``states[0]`` is the lstm hidden states, of a batch of utterance.
        ``states[1]`` is the lstm cell states, of a batch of utterances.

    Returns:
      A list of states.
        ``states[i]`` is a tuple of 2 elememts of i-th utterance.
        ``states[i][0]`` is the lstm hidden states of i-th utterance.
        ``states[i][1]`` is the lstm cell states of i-th utterance.
    """
    hidden_states, cell_states = states

    list_hidden_states = hidden_states.unbind(dim=1)
    list_cell_states = cell_states.unbind(dim=1)

    ans = [
        (h.unsqueeze(1), c.unsqueeze(1))
        for (h, c) in zip(list_hidden_states, list_cell_states)
    ]
    return ans


def stack_states(
    states_list: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack list of lstm states corresponding to separate utterances into a single
    lstm state so that it can be used as an input for lstm when those utterances
    are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponds to the lstm state for a single
        utterance.
        ``states[i]`` is a tuple of 2 elememts of i-th utterance.
        ``states[i][0]`` is the lstm hidden states of i-th utterance.
        ``states[i][1]`` is the lstm cell states of i-th utterance.


    Returns:
      A new state corresponding to a batch of utterances.
      It is a tuple of 2 elements.
        ``states[0]`` is the lstm hidden states, of a batch of utterance.
        ``states[1]`` is the lstm cell states, of a batch of utterances.
    """
    hidden_states = torch.cat([s[0] for s in states_list], dim=1)
    cell_states = torch.cat([s[1] for s in states_list], dim=1)
    ans = (hidden_states, cell_states)
    return ans


class RNN(EncoderInterface):
    """
    Args:
      num_features (int):
        Number of input features.
      subsampling_factor (int):
        Subsampling factor of encoder (convolution layers before lstm layers) (default=4).  # noqa
      d_model (int):
        Output dimension (default=512).
      dim_feedforward (int):
        Feedforward dimension (default=2048).
      rnn_hidden_size (int):
        Hidden dimension for lstm layers (default=1024).
      num_encoder_layers (int):
        Number of encoder layers (default=12).
      dropout (float):
        Dropout rate (default=0.1).
      layer_dropout (float):
        Dropout value for model-level warmup (default=0.075).
      aux_layer_period (int):
        Period of auxiliary layers used for random combiner during training.
        If set to 0, will not use the random combiner (Default).
        You can set a positive integer to use the random combiner, e.g., 3.
      is_pnnx:
        True to make this class exportable via PNNX.
    """

    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        rnn_hidden_size: int = 1024,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        aux_layer_period: int = 0,
        is_pnnx: bool = False,
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
        self.encoder_embed = Conv2dSubsampling(
            num_features,
            d_model,
            is_pnnx=is_pnnx,
        )

        self.is_pnnx = is_pnnx

        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.rnn_hidden_size = rnn_hidden_size

        encoder_layer = RNNEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            rnn_hidden_size=rnn_hidden_size,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )
        self.encoder = RNNEncoder(
            encoder_layer,
            num_encoder_layers,
            aux_layers=list(
                range(
                    num_encoder_layers // 3,
                    num_encoder_layers - 1,
                    aux_layer_period,
                )
            )
            if aux_layer_period > 0
            else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, T, C), where N is the batch size,
            T is the sequence length, C is the feature dimension.
          x_lens:
            A tensor of shape (N,), containing the number of frames in `x`
            before padding.
          states:
            A tuple of 2 tensors (optional). It is for streaming inference.
            states[0] is the hidden states of all layers,
              with shape of (num_layers, N, d_model);
            states[1] is the cell states of all layers,
              with shape of (num_layers, N, rnn_hidden_size).
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          A tuple of 3 tensors:
            - embeddings: its shape is (N, T', d_model), where T' is the output
              sequence lengths.
            - lengths: a tensor of shape (batch_size,) containing the number of
              frames in `embeddings` before padding.
            - updated states, whose shape is the same as the input states.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # lengths = ((x_lens - 3) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        if not self.is_pnnx:
            lengths = (((x_lens - 3) >> 1) - 1) >> 1
        else:
            lengths1 = torch.floor((x_lens - 3) / 2)
            lengths = torch.floor((lengths1 - 1) / 2)
            lengths = lengths.to(x_lens)

        if not torch.jit.is_tracing():
            assert x.size(0) == lengths.max().item()

        if states is None:
            x = self.encoder(x, warmup=warmup)[0]
            # torch.jit.trace requires returned types to be the same as annotated  # noqa
            new_states = (torch.empty(0), torch.empty(0))
        else:
            assert not self.training
            assert len(states) == 2
            if not torch.jit.is_tracing():
                # for hidden state
                assert states[0].shape == (
                    self.num_encoder_layers,
                    x.size(1),
                    self.d_model,
                )
                # for cell state
                assert states[1].shape == (
                    self.num_encoder_layers,
                    x.size(1),
                    self.rnn_hidden_size,
                )
            x, new_states = self.encoder(x, states)

        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        return x, lengths, new_states

    @torch.jit.export
    def get_init_states(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get model initial states."""
        # for rnn hidden states
        hidden_states = torch.zeros(
            (self.num_encoder_layers, batch_size, self.d_model), device=device
        )
        cell_states = torch.zeros(
            (self.num_encoder_layers, batch_size, self.rnn_hidden_size),
            device=device,
        )
        return (hidden_states, cell_states)


class RNNEncoderLayer(nn.Module):
    """
    RNNEncoderLayer is made up of lstm and feedforward networks.

    Args:
      d_model:
        The number of expected features in the input (required).
      dim_feedforward:
        The dimension of feedforward network model (default=2048).
      rnn_hidden_size:
        The hidden dimension of rnn layer.
      dropout:
        The dropout value (default=0.1).
      layer_dropout:
        The dropout value for model-level warmup (default=0.075).
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        rnn_hidden_size: int,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ) -> None:
        super(RNNEncoderLayer, self).__init__()
        self.layer_dropout = layer_dropout
        self.d_model = d_model
        self.rnn_hidden_size = rnn_hidden_size

        assert rnn_hidden_size >= d_model, (rnn_hidden_size, d_model)
        self.lstm = ScaledLSTM(
            input_size=d_model,
            hidden_size=rnn_hidden_size,
            proj_size=d_model if rnn_hidden_size > d_model else 0,
            num_layers=1,
            dropout=0.0,
        )
        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )
        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).  # noqa
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pass the input through the encoder layer.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          states:
            A tuple of 2 tensors (optional). It is for streaming inference.
            states[0] is the hidden states of all layers,
              with shape of (1, N, d_model);
            states[1] is the cell states of all layers,
              with shape of (1, N, rnn_hidden_size).
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
        if states is None:
            src_lstm = self.lstm(src)[0]
            # torch.jit.trace requires returned types be the same as annotated
            new_states = (torch.empty(0), torch.empty(0))
        else:
            assert not self.training
            assert len(states) == 2
            if not torch.jit.is_tracing():
                # for hidden state
                assert states[0].shape == (1, src.size(1), self.d_model)
                # for cell state
                assert states[1].shape == (1, src.size(1), self.rnn_hidden_size)
            src_lstm, new_states = self.lstm(src, states)
        src = self.dropout(src_lstm) + src

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

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

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        aux_layers: Optional[List[int]] = None,
    ) -> None:
        super(RNNEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = encoder_layer.d_model
        self.rnn_hidden_size = encoder_layer.rnn_hidden_size

        self.aux_layers: List[int] = []
        self.combiner: Optional[nn.Module] = None
        if aux_layers is not None:
            assert len(set(aux_layers)) == len(aux_layers)
            assert num_layers - 1 not in aux_layers
            self.aux_layers = aux_layers + [num_layers - 1]
            self.combiner = RandomCombine(
                num_inputs=len(self.aux_layers),
                final_weight=0.5,
                pure_prob=0.333,
                stddev=2.0,
            )

    def forward(
        self,
        src: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pass the input through the encoder layer in turn.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          states:
            A tuple of 2 tensors (optional). It is for streaming inference.
            states[0] is the hidden states of all layers,
              with shape of (num_layers, N, d_model);
            states[1] is the cell states of all layers,
              with shape of (num_layers, N, rnn_hidden_size).
          warmup:
            It controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.
        """
        if states is not None:
            assert not self.training
            assert len(states) == 2
            if not torch.jit.is_tracing():
                # for hidden state
                assert states[0].shape == (
                    self.num_layers,
                    src.size(1),
                    self.d_model,
                )
                # for cell state
                assert states[1].shape == (
                    self.num_layers,
                    src.size(1),
                    self.rnn_hidden_size,
                )

        output = src

        outputs = []

        new_hidden_states = []
        new_cell_states = []

        for i, mod in enumerate(self.layers):
            if states is None:
                output = mod(output, warmup=warmup)[0]
            else:
                layer_state = (
                    states[0][i : i + 1, :, :],  # h: (1, N, d_model)
                    states[1][i : i + 1, :, :],  # c: (1, N, rnn_hidden_size)
                )
                output, (h, c) = mod(output, layer_state)
                new_hidden_states.append(h)
                new_cell_states.append(c)

            if self.combiner is not None and i in self.aux_layers:
                outputs.append(output)

        if self.combiner is not None:
            output = self.combiner(outputs)

        if states is None:
            new_states = (torch.empty(0), torch.empty(0))
        else:
            new_states = (
                torch.cat(new_hidden_states, dim=0),
                torch.cat(new_cell_states, dim=0),
            )

        return output, new_states


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-3)//2-1)//2, which approximates T' == T//4

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
            Caution: It requires: T >= 9, in_channels >= 9.
          out_channels
            Output dim. The output shape is (N, ((T-3)//2-1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
          is_pnnx:
            True if we are converting the model to PNNX format.
            False otherwise.
        """
        assert in_channels >= 9
        super().__init__()

        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=0,
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
            layer3_channels * (((in_channels - 3) // 2 - 1) // 2), out_channels
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
          Return a tensor of shape (N, ((T-3)//2-1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)

        if torch.jit.is_tracing() and self.is_pnnx:
            x = x.permute(0, 2, 1, 3).reshape(1, -1, self.conv_out_dim)
            x = self.out(x)
        else:
            # Now x is of shape (N, odim, ((T-3)//2-1)//2, ((idim-3)//2-1)//2)
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        # Now x is of shape (N, ((T-3)//2-1))//2, odim)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x


class RandomCombine(nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.

    The idea is that the list of Tensors will be a list of outputs of multiple
    conformer layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        num_inputs: int,
        final_weight: float = 0.5,
        pure_prob: float = 0.5,
        stddev: float = 2.0,
    ) -> None:
        """
        Args:
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          final_weight:
            The amount of weight or probability we assign to the
            final layer when randomly choosing layers or when choosing
            continuous layer weights.
          pure_prob:
            The probability, on each frame, with which we choose
            only a single layer to output (rather than an interpolation)
          stddev:
            A standard deviation that we add to log-probs for computing
            randomized weights.

        The method of choosing which layers, or combinations of layers, to use,
        is conceptually as follows::

            With probability `pure_prob`::
               With probability `final_weight`: choose final layer,
               Else: choose random non-final layer.
            Else::
               Choose initial log-weights that correspond to assigning
               weight `final_weight` to the final layer and equal
               weights to other layers; then add Gaussian noise
               with variance `stddev` to these log-weights, and normalize
               to weights (note: the average weight assigned to the
               final layer here will not be `final_weight` if stddev>0).
        """
        super().__init__()
        assert 0 <= pure_prob <= 1, pure_prob
        assert 0 < final_weight < 1, final_weight
        assert num_inputs >= 1

        self.num_inputs = num_inputs
        self.final_weight = final_weight
        self.pure_prob = pure_prob
        self.stddev = stddev

        self.final_log_weight = (
            torch.tensor((final_weight / (1 - final_weight)) * (self.num_inputs - 1))
            .log()
            .item()
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels).  In test mode
          this is just the final input.
        """
        num_inputs = self.num_inputs
        assert len(inputs) == num_inputs
        if not self.training or torch.jit.is_scripting():
            return inputs[-1]

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        # weights: (num_frames, num_inputs)
        weights = self._get_random_weights(
            inputs[0].dtype, inputs[0].device, num_frames
        )

        weights = weights.reshape(num_frames, num_inputs, 1)
        # ans: (num_frames, num_channels, 1)
        ans = torch.matmul(stacked_inputs, weights)
        # ans: (*, num_channels)

        ans = ans.reshape(inputs[0].shape[:-1] + (num_channels,))

        # The following if causes errors for torch script in torch 1.6.0
        #  if __name__ == "__main__":
        #      # for testing only...
        #      print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans

    def _get_random_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ) -> torch.Tensor:
        """Return a tensor of random weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired
        Returns:
          A tensor of shape (num_frames, self.num_inputs), such that
          `ans.sum(dim=1)` is all ones.
        """
        pure_prob = self.pure_prob
        if pure_prob == 0.0:
            return self._get_random_mixed_weights(dtype, device, num_frames)
        elif pure_prob == 1.0:
            return self._get_random_pure_weights(dtype, device, num_frames)
        else:
            p = self._get_random_pure_weights(dtype, device, num_frames)
            m = self._get_random_mixed_weights(dtype, device, num_frames)
            return torch.where(
                torch.rand(num_frames, 1, device=device) < self.pure_prob, p, m
            )

    def _get_random_pure_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ):
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A one-hot tensor of shape `(num_frames, self.num_inputs)`, with
          exactly one weight equal to 1.0 on each frame.
        """
        final_prob = self.final_weight

        # final contains self.num_inputs - 1 in all elements
        final = torch.full((num_frames,), self.num_inputs - 1, device=device)
        # nonfinal contains random integers in [0..num_inputs - 2], these are for non-final weights.  # noqa
        nonfinal = torch.randint(self.num_inputs - 1, (num_frames,), device=device)

        indexes = torch.where(
            torch.rand(num_frames, device=device) < final_prob, final, nonfinal
        )
        ans = torch.nn.functional.one_hot(indexes, num_classes=self.num_inputs).to(
            dtype=dtype
        )
        return ans

    def _get_random_mixed_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ):
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A tensor of shape (num_frames, self.num_inputs), which elements
          in [0..1] that sum to one over the second axis, i.e.
          `ans.sum(dim=1)` is all ones.
        """
        logprobs = (
            torch.randn(num_frames, self.num_inputs, dtype=dtype, device=device)
            * self.stddev  # noqa
        )
        logprobs[:, -1] += self.final_log_weight
        return logprobs.softmax(dim=1)


def _test_random_combine(final_weight: float, pure_prob: float, stddev: float):
    print(
        f"_test_random_combine: final_weight={final_weight}, pure_prob={pure_prob}, stddev={stddev}"  # noqa
    )
    num_inputs = 3
    num_channels = 50
    m = RandomCombine(
        num_inputs=num_inputs,
        final_weight=final_weight,
        pure_prob=pure_prob,
        stddev=stddev,
    )

    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_random_combine_main():
    _test_random_combine(0.999, 0, 0.0)
    _test_random_combine(0.5, 0, 0.0)
    _test_random_combine(0.999, 0, 0.0)
    _test_random_combine(0.5, 0, 0.3)
    _test_random_combine(0.5, 1, 0.3)
    _test_random_combine(0.5, 0.5, 0.3)

    feature_dim = 50
    c = RNN(num_features=feature_dim, d_model=128)
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    feature_dim = 80
    m = RNN(
        num_features=feature_dim,
        d_model=512,
        rnn_hidden_size=1024,
        dim_feedforward=2048,
        num_encoder_layers=12,
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = m(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        warmup=0.5,
    )
    num_param = sum([p.numel() for p in m.parameters()])
    print(f"Number of model parameters: {num_param}")

    _test_random_combine_main()
