# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

"""
Apply layer normalization to the output of each gate in LSTM/GRU.

This file uses
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
as a reference.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO(fangjun): Support projection, see https://arxiv.org/pdf/1402.1128.pdf
class LayerNormLSTMCell(nn.Module):
    """This class places a `nn.LayerNorm` after the output of
    each gate (right before the activation).

    See the following paper for more details

    'Improving RNN Transducer Modeling for End-to-End Speech Recognition'
    https://arxiv.org/abs/1909.12415
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        ln: nn.Module = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        """
        Args:
          input_size:
            The number of expected features in the input `x`. `x` should
            be of shape (batch_size, input_size).
          hidden_size:
            The number of features in the hidden state `h` and `c`.
            Both `h` and `c` are of shape (batch_size, hidden_size).
          bias:
            If ``False``, then the cell does not use bias weights
            `bias_ih` and `bias_hh`.
          ln:
            Defaults to `nn.LayerNorm`. The output of all gates are processed
            by `ln`. We pass it as an argument so that we can replace it
            with `nn.Identity` at the testing time.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = nn.Parameter(
            torch.empty((4 * hidden_size, input_size), **factory_kwargs)
        )

        self.weight_hh = nn.Parameter(
            torch.empty((4 * hidden_size, hidden_size), **factory_kwargs)
        )

        if bias:
            self.bias_ih = nn.Parameter(
                torch.empty(4 * hidden_size, **factory_kwargs)
            )
            self.bias_hh = nn.Parameter(
                torch.empty(4 * hidden_size, **factory_kwargs)
            )
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.layernorm_i = ln(hidden_size)
        self.layernorm_f = ln(hidden_size)
        self.layernorm_cx = ln(hidden_size)
        self.layernorm_cy = ln(hidden_size)
        self.layernorm_o = ln(hidden_size)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          input:
            A 2-D tensor of shape (batch_size, input_size).
          state:
            If not ``None``, it contains the hidden state (h, c); both
            are of shape (batch_size, hidden_size). If ``None``, it uses
            zeros for `h` and `c`.
        Returns:
          Return two tensors:
            - `next_h`: It is of shape (batch_size, hidden_size) containing the
              next hidden state for each element in the batch.
            - `next_c`: It is of shape (batch_size, hidden_size) containing the
              next cell state for each element in the batch.
        """
        if state is None:
            zeros = torch.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            state = (zeros, zeros)

        hx, cx = state
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(
            hx, self.weight_hh, self.bias_hh
        )

        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(chunks=4, dim=1)

        in_gate = self.layernorm_i(in_gate)
        forget_gate = self.layernorm_f(forget_gate)
        cell_gate = self.layernorm_cx(cell_gate)
        out_gate = self.layernorm_o(out_gate)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        cy = self.layernorm_cy(cy)
        hy = out_gate * torch.tanh(cy)

        return hy, cy

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


class LayerNormLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        ln: nn.Module = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        """
        See the args in LayerNormLSTMCell
        """
        super().__init__()
        self.cell = LayerNormLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            ln=ln,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          input:
            A 3-D tensor of shape (batch_size, seq_len, input_size).
            Caution:
              We use `batch_first=True` here.
          state:
            If not ``None``, it contains the hidden state (h, c) of this layer.
            Both are of shape (batch_size, hidden_size).
            Note:
              We did not annotate `state` with `Optional[Tuple[...]]` since
              torchscript will complain.
        Return:
          - output, a tensor of shape (batch_size, seq_len, hidden_size)
          - (next_h, next_c) containing the hidden state of this layer
        """
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state[0]]
        return torch.stack(outputs, dim=1), state


class LayerNormLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        ln: nn.Module = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        """
        See the args in LSTMLayer.
        """
        super().__init__()
        assert num_layers >= 1
        factory_kwargs = dict(
            hidden_size=hidden_size,
            bias=bias,
            ln=ln,
            device=device,
            dtype=dtype,
        )
        first_layer = LayerNormLSTMLayer(
            input_size=input_size, **factory_kwargs
        )
        layers = [first_layer]
        for i in range(1, num_layers):
            layers.append(
                LayerNormLSTMLayer(
                    input_size=hidden_size,
                    **factory_kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(
        self,
        input: torch.Tensor,
        states: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
          input:
            A 3-D tensor of shape (batch_size, seq_len, input_size).
            Caution:
              We use `batch_first=True` here.
          states:
            One state per layer. Each entry contains the hidden state (h, c)
            for a layer. Both are of shape (batch_size, hidden_size).
        Returns:
          Return a tuple containing:

            - output: A tensor of shape (batch_size, seq_len, hidden_size)
            - List[(next_h, next_c)] containing the hidden states for all layers

        """
        output_states = torch.jit.annotate(
            List[Tuple[torch.Tensor, torch.Tensor]], []
        )
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states
