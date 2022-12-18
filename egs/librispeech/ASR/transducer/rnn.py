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
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types


class LayerNormLSTMCell(nn.Module):
    """This class places a `nn.LayerNorm` after the output of
    each gate (right before the activation).

    See the following paper for more details

    'Improving RNN Transducer Modeling for End-to-End Speech Recognition'
    https://arxiv.org/abs/1909.12415

    Examples::

        >>> cell = LayerNormLSTMCell(10, 20)
        >>> input = torch.rand(5, 10)
        >>> h0 = torch.rand(5, 20)
        >>> c0 = torch.rand(5, 20)
        >>> h1, c1 = cell(input, (h0, c0))
        >>> output = h1
        >>> h1.shape
        torch.Size([5, 20])
        >>> c1.shape
        torch.Size([5, 20])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        ln: Type[nn.Module] = nn.LayerNorm,
        proj_size: int = 0,
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
            Both `h` and `c` are of shape (batch_size, hidden_size) when
            proj_size is 0. If proj_size is not zero, the shape of `h`
            is (batch_size, proj_size).
          bias:
            If ``False``, then the cell does not use bias weights
            `bias_ih` and `bias_hh`.
          ln:
            Defaults to `nn.LayerNorm`. The output of all gates are processed
            by `ln`. We pass it as an argument so that we can replace it
            with `nn.Identity` at the testing time.
          proj_size:
            If not zero, it applies an affine transform to the output. In this
            case, the shape of `h` is (batch_size, proj_size).
            See https://arxiv.org/pdf/1402.1128.pdf
        """
        assert check_argument_types()
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.proj_size = proj_size

        if proj_size < 0:
            raise ValueError(
                f"proj_size {proj_size} should be a positive integer "
                "or zero to disable projections"
            )

        if proj_size >= hidden_size:
            raise ValueError(
                f"proj_size {proj_size} has to be smaller "
                f"than hidden_size {hidden_size}"
            )

        real_hidden_size = proj_size if proj_size > 0 else hidden_size

        self.weight_ih = nn.Parameter(
            torch.empty((4 * hidden_size, input_size), **factory_kwargs)
        )

        self.weight_hh = nn.Parameter(
            torch.empty((4 * hidden_size, real_hidden_size), **factory_kwargs)
        )

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size, **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        if proj_size > 0:
            self.weight_hr = nn.Parameter(
                torch.empty((proj_size, hidden_size), **factory_kwargs)
            )
        else:
            self.register_parameter("weight_hr", None)

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
            If not ``None``, it contains the hidden state (h, c) for each
            element in the batch. Both are of shape (batch_size, hidden_size)
            if proj_size is 0. If proj_size is not zero, the shape of `h` is
            (batch_size, proj_size).
            If ``None``, it uses zeros for `h` and `c`.
        Returns:
          Return two tensors:
            - `next_h`: It is of shape (batch_size, hidden_size) if proj_size
              is 0, else (batch_size, proj_size), containing the next hidden
              state for each element in the batch.
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

        if self.weight_hr is not None:
            hy = torch.matmul(hy, self.weight_hr.t())

        return hy, cy

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if "layernorm" not in name:
                nn.init.uniform_(weight, -stdv, stdv)

            if "bias_ih" in name or "bias_hh" in name:
                # See the paper
                # An Empirical Exploration of Recurrent Network Architectures
                # https://proceedings.mlr.press/v37/jozefowicz15.pdf
                #
                # It recommends initializing the bias of the forget gate to
                # a large value, such as 1 or 2. In PyTorch, there are two
                # biases for the forget gate, we set both of them to 1 here.
                #
                # See also https://arxiv.org/pdf/1804.04849.pdf
                assert weight.ndim == 1
                # Layout of the bias:
                # | in_gate | forget_gate | cell_gate | output_gate |
                start = weight.numel() // 4
                end = weight.numel() // 2
                with torch.no_grad():
                    weight[start:end].fill_(1.0)


class LayerNormLSTMLayer(nn.Module):
    """
    Examples::

        >>> layer = LayerNormLSTMLayer(10, 20)
        >>> input = torch.rand(2, 5, 10)
        >>> h0 = torch.rand(2, 20)
        >>> c0 = torch.rand(2, 20)
        >>> output, (hn, cn) = layer(input, (h0, c0))
        >>> output.shape
        torch.Size([2, 5, 20])
        >>> hn.shape
        torch.Size([2, 20])
        >>> cn.shape
        torch.Size([2, 20])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        ln: Type[nn.Module] = nn.LayerNorm,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ):
        """
        See the args in LayerNormLSTMCell
        """
        assert check_argument_types()
        super().__init__()
        self.cell = LayerNormLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            ln=ln,
            proj_size=proj_size,
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
            Both are of shape (batch_size, hidden_size) if proj_size is 0.
            If proj_size is not 0, the shape of `h` is (batch_size, proj_size).
            Note:
              We did not annotate `state` with `Optional[Tuple[...]]` since
              torchscript will complain.
        Return:
          - output, a tensor of shape (batch_size, seq_len, hidden_size)
          - (next_h, next_c) containing the next hidden state
        """
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs.append(state[0])
        return torch.stack(outputs, dim=1), state


class LayerNormLSTM(nn.Module):
    """
    Examples::

        >>> lstm = LayerNormLSTM(10, 20, 8)
        >>> input = torch.rand(2, 3, 10)
        >>> h0 = torch.rand(8, 2, 20).unbind(0)
        >>> c0 = torch.rand(8, 2, 20).unbind(0)
        >>> states = list(zip(h0, c0))
        >>> output, next_states = lstm(input, states)
        >>> output.shape
        torch.Size([2, 3, 20])
        >>> hn = torch.stack([s[0] for s in next_states])
        >>> cn = torch.stack([s[1] for s in next_states])
        >>> hn.shape
        torch.Size([8, 2, 20])
        >>> cn.shape
        torch.Size([8, 2, 20])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        proj_size: int = 0,
        ln: Type[nn.Module] = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        """
        See the args in LayerNormLSTMLayer.
        """
        assert check_argument_types()
        super().__init__()
        assert num_layers >= 1
        factory_kwargs = dict(
            hidden_size=hidden_size,
            bias=bias,
            ln=ln,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )
        first_layer = LayerNormLSTMLayer(input_size=input_size, **factory_kwargs)
        layers = [first_layer]
        for i in range(1, num_layers):
            layers.append(
                LayerNormLSTMLayer(
                    input_size=proj_size if proj_size > 0 else hidden_size,
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
            for a layer. Both are of shape (batch_size, hidden_size) if
            proj_size is 0. If proj_size is not 0, the shape of `h` is
            (batch_size, proj_size).
        Returns:
          Return a tuple containing:

            - output: A tensor of shape (batch_size, seq_len, hidden_size)
            - List[(next_h, next_c)] containing the hidden states for all layers

        """
        output_states = torch.jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states


class LayerNormGRUCell(nn.Module):
    """This class places a `nn.LayerNorm` after the output of
    each gate (right before the activation).

    See the following paper for more details

    'Improving RNN Transducer Modeling for End-to-End Speech Recognition'
    https://arxiv.org/abs/1909.12415

    Examples::

        >>> cell = LayerNormGRUCell(10, 20)
        >>> input = torch.rand(2, 10)
        >>> h0 = torch.rand(2, 20)
        >>> hn = cell(input, h0)
        >>> hn.shape
        torch.Size([2, 20])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        ln: Type[nn.Module] = nn.LayerNorm,
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
        assert check_argument_types()
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = nn.Parameter(
            torch.empty((3 * hidden_size, input_size), **factory_kwargs)
        )

        self.weight_hh = nn.Parameter(
            torch.empty((3 * hidden_size, hidden_size), **factory_kwargs)
        )

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size, **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.layernorm_r = ln(hidden_size)
        self.layernorm_i = ln(hidden_size)
        self.layernorm_n = ln(hidden_size)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          input:
            A 2-D tensor of shape (batch_size, input_size) containing
            input features.
          hx:
            If not `None`, it is a tensor of shape (batch_size, hidden_size)
            containing the initial hidden state for each element in the batch.
            If `None`, it uses zeros for the hidden state.
        Returns:
          Return a tensor of shape (batch_size, hidden_size) containing the
          next hidden state for each element in the batch
        """
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        i_r, i_i, i_n = F.linear(input, self.weight_ih, self.bias_ih).chunk(
            chunks=3, dim=1
        )

        h_r, h_i, h_n = F.linear(hx, self.weight_hh, self.bias_hh).chunk(
            chunks=3, dim=1
        )

        reset_gate = torch.sigmoid(self.layernorm_r(i_r + h_r))
        input_gate = torch.sigmoid(self.layernorm_i(i_i + h_i))
        new_gate = torch.tanh(self.layernorm_n(i_n + reset_gate * h_n))

        # hy = (1 - input_gate) * new_gate + input_gate * hx
        #    = new_gate - input_gate * new_gate + input_gate * hx
        #    = new_gate + input_gate * (hx - new_gate)
        hy = new_gate + input_gate * (hx - new_gate)

        return hy

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


class LayerNormGRULayer(nn.Module):
    """
    Examples::

        >>> layer = LayerNormGRULayer(10, 20)
        >>> input = torch.rand(2, 3, 10)
        >>> hx = torch.rand(2, 20)
        >>> output, hn = layer(input, hx)
        >>> output.shape
        torch.Size([2, 3, 20])
        >>> hn.shape
        torch.Size([2, 20])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        ln: Type[nn.Module] = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        """
        See the args in LayerNormGRUCell
        """
        assert check_argument_types()
        super().__init__()
        self.cell = LayerNormGRUCell(
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
        hx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          input:
            A 3-D tensor of shape (batch_size, seq_len, input_size).
            Caution:
              We use `batch_first=True` here.
          hx:
            If not ``None``, it is a tensor of shape (batch_size, hidden_size)
            containing the hidden state for each element in the batch.
        Return:
          - output, a tensor of shape (batch_size, seq_len, hidden_size)
          - next_h, a tensor of shape (batch_size, hidden_size) containing the
            final hidden state for each element in the batch.
        """
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        next_h = hx
        for i in range(len(inputs)):
            next_h = self.cell(inputs[i], next_h)
            outputs.append(next_h)
        return torch.stack(outputs, dim=1), next_h


class LayerNormGRU(nn.Module):
    """
    Examples::

        >>> input = torch.rand(2, 3, 10)
        >>> h0 = torch.rand(8, 2, 20)
        >>> states = h0.unbind(0)
        >>> output, next_states = gru(input, states)
        >>> output.shape
        torch.Size([2, 3, 20])
        >>> hn = torch.stack(next_states)
        >>> hn.shape
        torch.Size([8, 2, 20])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        ln: Type[nn.Module] = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        """
        See the args in LayerNormGRULayer.
        """
        assert check_argument_types()
        super().__init__()
        assert num_layers >= 1
        factory_kwargs = dict(
            hidden_size=hidden_size,
            bias=bias,
            ln=ln,
            device=device,
            dtype=dtype,
        )
        first_layer = LayerNormGRULayer(input_size=input_size, **factory_kwargs)
        layers = [first_layer]
        for i in range(1, num_layers):
            layers.append(
                LayerNormGRULayer(
                    input_size=hidden_size,
                    **factory_kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(
        self,
        input: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
          input:
            A tensor of shape (batch_size, seq_len, input_size) containing
            input features.
            Caution:
              We use `batch_first=True` here.
          states:
            One state per layer. Each entry contains the hidden state for each
            element in the batch. Each hidden state is of shape
            (batch_size, hidden_size)
        Returns:
          Return a tuple containing:

            - output: A tensor of shape (batch_size, seq_len, hidden_size)
            - List[next_state] containing the final hidden states for each
              element in the batch

        """
        output_states = torch.jit.annotate(List[torch.Tensor], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states
