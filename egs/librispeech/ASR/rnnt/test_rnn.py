#!/usr/bin/env python3
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

import torch
import torch.nn as nn
from rnnt.rnn import LayerNormLSTM, LayerNormLSTMCell, LayerNormLSTMLayer


def test_layernorm_lstm_cell_jit():
    input_size = 10
    hidden_size = 20
    cell = LayerNormLSTMCell(
        input_size=input_size, hidden_size=hidden_size, bias=True
    )

    torch.jit.script(cell)


def test_layernorm_lstm_cell_constructor():
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()

    self_cell = LayerNormLSTMCell(input_size, hidden_size, ln=nn.Identity)
    torch_cell = nn.LSTMCell(input_size, hidden_size)

    for name, param in self_cell.named_parameters():
        assert param.shape == getattr(torch_cell, name).shape

    assert len(self_cell.state_dict()) == len(torch_cell.state_dict())


def test_layernorm_lstm_cell_forward():
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0

    self_cell = LayerNormLSTMCell(
        input_size, hidden_size, bias=bias, ln=nn.Identity
    )
    torch_cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
    with torch.no_grad():
        for name, torch_param in torch_cell.named_parameters():
            self_param = getattr(self_cell, name)
            torch_param.copy_(self_param)

    N = torch.randint(low=2, high=100, size=(1,))
    x = torch.rand(N, input_size).requires_grad_()
    h = torch.rand(N, hidden_size)
    c = torch.rand(N, hidden_size)

    x_clone = x.detach().clone().requires_grad_()

    self_h, self_c = self_cell(x.clone(), (h, c))
    torch_h, torch_c = torch_cell(x_clone, (h, c))

    assert torch.allclose(self_h, torch_h)
    assert torch.allclose(self_c, torch_c)

    self_hc = self_h * self_c
    torch_hc = torch_h * torch_c
    (self_hc.reshape(-1) * torch.arange(self_hc.numel())).sum().backward()
    (torch_hc.reshape(-1) * torch.arange(torch_hc.numel())).sum().backward()

    assert torch.allclose(x.grad, x_clone.grad)


def test_lstm_layer_jit():
    input_size = 10
    hidden_size = 20
    layer = LayerNormLSTMLayer(input_size, hidden_size=hidden_size)
    torch.jit.script(layer)


def test_lstm_layer_forward():
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0
    self_layer = LayerNormLSTMLayer(
        input_size,
        hidden_size,
        bias=bias,
        ln=nn.Identity,
    )

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size).requires_grad_()
    h = torch.rand(N, hidden_size)
    c = torch.rand(N, hidden_size)

    x_clone = x.detach().clone().requires_grad_()

    self_y, (self_h, self_c) = self_layer(x, (h, c))

    # now for pytorch
    torch_layer = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=bias,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )
    with torch.no_grad():
        for name, self_param in self_layer.cell.named_parameters():
            getattr(torch_layer, f"{name}_l0").copy_(self_param)

    torch_y, (torch_h, torch_c) = torch_layer(
        x_clone, (h.unsqueeze(0), c.unsqueeze(0))
    )
    assert torch.allclose(self_y, torch_y)
    assert torch.allclose(self_h, torch_h)
    assert torch.allclose(self_c, torch_c)

    self_hc = self_h * self_c
    torch_hc = torch_h * torch_c
    self_hc_sum = (self_hc.reshape(-1) * torch.arange(self_hc.numel())).sum()
    torch_hc_sum = (torch_hc.reshape(-1) * torch.arange(torch_hc.numel())).sum()

    self_y_sum = (self_y.reshape(-1) * torch.arange(self_y.numel())).sum()
    torch_y_sum = (torch_y.reshape(-1) * torch.arange(torch_y.numel())).sum()

    (self_hc_sum * self_y_sum).backward()
    (torch_hc_sum * torch_y_sum).backward()

    assert torch.allclose(x.grad, x_clone.grad, rtol=0.1)


def test_stacked_lstm_jit():
    input_size = 2
    hidden_size = 3
    num_layers = 4
    bias = True

    lstm = LayerNormLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        ln=nn.Identity,
    )
    torch.jit.script(lstm)


def test_stacked_lstm_forward():
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    num_layers = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0

    self_lstm = LayerNormLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        ln=nn.Identity,
    )
    torch_lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=True,
        bidirectional=False,
    )
    assert len(self_lstm.state_dict()) == len(torch_lstm.state_dict())
    with torch.no_grad():
        for name, param in self_lstm.named_parameters():
            # name has the form layers.0.cell.weight_hh
            parts = name.split(".")
            layer_num = parts[1]
            getattr(torch_lstm, f"{parts[-1]}_l{layer_num}").copy_(param)

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size).requires_grad_()
    hs = [torch.rand(N, hidden_size) for _ in range(num_layers)]
    cs = [torch.rand(N, hidden_size) for _ in range(num_layers)]
    states = list(zip(hs, cs))

    x_clone = x.detach().clone().requires_grad_()

    self_y, self_states = self_lstm(x, states)

    h = torch.stack(hs)
    c = torch.stack(cs)
    torch_y, (torch_h, torch_c) = torch_lstm(x_clone, (h, c))

    assert torch.allclose(self_y, torch_y)

    self_h = torch.stack([s[0] for s in self_states])
    self_c = torch.stack([s[1] for s in self_states])

    assert torch.allclose(self_h, torch_h)
    assert torch.allclose(self_c, torch_c)

    s = self_y.reshape(-1)
    t = torch_y.reshape(-1)

    s_sum = (s * torch.arange(s.numel())).sum()
    t_sum = (t * torch.arange(t.numel())).sum()
    shc_sum = s_sum * self_h.sum() * self_c.sum()
    thc_sum = t_sum * torch_h.sum() * torch_c.sum()

    shc_sum.backward()
    thc_sum.backward()
    assert torch.allclose(x.grad, x_clone.grad)


def main():
    test_layernorm_lstm_cell_jit()
    test_layernorm_lstm_cell_constructor()
    test_layernorm_lstm_cell_forward()
    #
    test_lstm_layer_jit()
    test_lstm_layer_forward()
    #
    test_stacked_lstm_jit()
    test_stacked_lstm_forward()


if __name__ == "__main__":
    main()
