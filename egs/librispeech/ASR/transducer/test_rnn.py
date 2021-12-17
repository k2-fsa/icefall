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
from transducer.rnn import (
    LayerNormGRU,
    LayerNormGRUCell,
    LayerNormGRULayer,
    LayerNormLSTM,
    LayerNormLSTMCell,
    LayerNormLSTMLayer,
)


def get_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", 0))
    return devices


def assert_allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-6, **kwargs):
    assert torch.allclose(
        a, b, atol=atol, **kwargs
    ), f"{(a - b).abs().max()}, {a.numel()}"


def test_layernorm_lstm_cell_jit(device="cpu"):
    input_size = 10
    hidden_size = 20
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0
    cell = LayerNormLSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        device=device,
    )

    torch.jit.script(cell)


def test_layernorm_lstm_cell_constructor(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()

    self_cell = LayerNormLSTMCell(
        input_size,
        hidden_size,
        ln=nn.Identity,
        device=device,
    )
    torch_cell = nn.LSTMCell(
        input_size,
        hidden_size,
    ).to(device)

    for name, param in self_cell.named_parameters():
        assert param.shape == getattr(torch_cell, name).shape

    assert len(self_cell.state_dict()) == len(torch_cell.state_dict())


def test_layernorm_lstm_cell_with_projection_jit(device="cpu"):
    input_size = 10
    hidden_size = 20
    proj_size = 5
    self_cell = LayerNormLSTMCell(
        input_size,
        hidden_size,
        proj_size=proj_size,
        device=device,
    )
    torch.jit.script(self_cell)


def test_layernorm_lstm_cell_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0

    self_cell = LayerNormLSTMCell(
        input_size,
        hidden_size,
        bias=bias,
        ln=nn.Identity,
        device=device,
    )
    torch_cell = nn.LSTMCell(
        input_size,
        hidden_size,
        bias=bias,
    ).to(device)
    with torch.no_grad():
        for name, torch_param in torch_cell.named_parameters():
            self_param = getattr(self_cell, name)
            torch_param.copy_(self_param)

    N = torch.randint(low=2, high=100, size=(1,))
    x = torch.rand(N, input_size, device=device).requires_grad_()
    h = torch.rand(N, hidden_size, device=device)
    c = torch.rand(N, hidden_size, device=device)

    x_clone = x.detach().clone().requires_grad_()

    self_h, self_c = self_cell(x.clone(), (h, c))
    torch_h, torch_c = torch_cell(x_clone, (h, c))

    assert_allclose(self_h, torch_h)
    assert_allclose(self_c, torch_c)

    self_hc = self_h * self_c
    torch_hc = torch_h * torch_c
    (
        self_hc.reshape(-1) * torch.arange(self_hc.numel(), device=device)
    ).sum().backward()
    (
        torch_hc.reshape(-1) * torch.arange(torch_hc.numel(), device=device)
    ).sum().backward()

    assert_allclose(x.grad, x_clone.grad, atol=1e-3)


def test_layernorm_lstm_cell_with_projection_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=10, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0
    proj_size = torch.randint(low=2, high=hidden_size, size=(1,)).item()

    self_cell = LayerNormLSTMCell(
        input_size,
        hidden_size,
        bias=bias,
        ln=nn.Identity,
        proj_size=proj_size,
        device=device,
    )
    torch_cell = nn.LSTM(
        input_size,
        hidden_size,
        bias=bias,
        proj_size=proj_size,
        batch_first=True,
    ).to(device)
    with torch.no_grad():
        for name, self_param in self_cell.named_parameters():
            getattr(torch_cell, f"{name}_l0").copy_(self_param)

    N = torch.randint(low=2, high=100, size=(1,))
    x = torch.rand(N, input_size, device=device).requires_grad_()
    h = torch.rand(N, proj_size, device=device)
    c = torch.rand(N, hidden_size, device=device)

    x_clone = x.detach().clone().requires_grad_()

    self_h, self_c = self_cell(x.clone(), (h, c))
    _, (torch_h, torch_c) = torch_cell(
        x_clone.unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0))
    )

    torch_h = torch_h.squeeze(0)
    torch_c = torch_c.squeeze(0)

    assert_allclose(self_h, torch_h)
    assert_allclose(self_c, torch_c)

    (self_h.sum() * self_c.sum()).backward()
    (torch_h.sum() * torch_c.sum()).backward()

    assert_allclose(x.grad, x_clone.grad, atol=1e-5)


def test_layernorm_lstm_layer_jit(device="cpu"):
    input_size = 10
    hidden_size = 20
    layer = LayerNormLSTMLayer(
        input_size,
        hidden_size=hidden_size,
        device=device,
    )
    torch.jit.script(layer)


def test_layernorm_lstm_layer_with_project_jit(device="cpu"):
    input_size = 10
    hidden_size = 20
    proj_size = 5
    layer = LayerNormLSTMLayer(
        input_size,
        hidden_size=hidden_size,
        proj_size=proj_size,
        device=device,
    )
    torch.jit.script(layer)


def test_layernorm_lstm_layer_with_projection_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=10, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0
    proj_size = torch.randint(low=2, high=hidden_size, size=(1,)).item()

    self_layer = LayerNormLSTMLayer(
        input_size,
        hidden_size,
        bias=bias,
        proj_size=proj_size,
        ln=nn.Identity,
        device=device,
    )

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size, device=device).requires_grad_()
    h = torch.rand(N, proj_size, device=device)
    c = torch.rand(N, hidden_size, device=device)

    x_clone = x.detach().clone().requires_grad_()

    self_y, (self_h, self_c) = self_layer(x, (h, c))

    torch_layer = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=bias,
        proj_size=proj_size,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    ).to(device)
    with torch.no_grad():
        for name, self_param in self_layer.cell.named_parameters():
            getattr(torch_layer, f"{name}_l0").copy_(self_param)

    torch_y, (torch_h, torch_c) = torch_layer(
        x_clone, (h.unsqueeze(0), c.unsqueeze(0))
    )
    assert_allclose(self_y, torch_y)
    assert_allclose(self_h, torch_h)
    assert_allclose(self_c, torch_c)

    self_y.sum().backward()
    torch_y.sum().backward()

    assert_allclose(x.grad, x_clone.grad, atol=1e-5)


def test_layernorm_lstm_layer_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0
    self_layer = LayerNormLSTMLayer(
        input_size,
        hidden_size,
        bias=bias,
        ln=nn.Identity,
        device=device,
    )

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size, device=device).requires_grad_()
    h = torch.rand(N, hidden_size, device=device)
    c = torch.rand(N, hidden_size, device=device)

    x_clone = x.detach().clone().requires_grad_()

    self_y, (self_h, self_c) = self_layer(x, (h, c))

    torch_layer = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=bias,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    ).to(device)
    with torch.no_grad():
        for name, self_param in self_layer.cell.named_parameters():
            getattr(torch_layer, f"{name}_l0").copy_(self_param)

    torch_y, (torch_h, torch_c) = torch_layer(
        x_clone, (h.unsqueeze(0), c.unsqueeze(0))
    )
    assert_allclose(self_y, torch_y)
    assert_allclose(self_h, torch_h)
    assert_allclose(self_c, torch_c)

    self_hc = self_h * self_c
    torch_hc = torch_h * torch_c
    self_hc_sum = (
        self_hc.reshape(-1) * torch.arange(self_hc.numel(), device=device)
    ).sum()
    torch_hc_sum = (
        torch_hc.reshape(-1) * torch.arange(torch_hc.numel(), device=device)
    ).sum()

    self_y_sum = (
        self_y.reshape(-1) * torch.arange(self_y.numel(), device=device)
    ).sum()
    torch_y_sum = (
        torch_y.reshape(-1) * torch.arange(torch_y.numel(), device=device)
    ).sum()

    (self_hc_sum + self_y_sum).backward()
    (torch_hc_sum + torch_y_sum).backward()

    assert_allclose(x.grad, x_clone.grad, atol=0.1)


def test_layernorm_lstm_jit(device="cpu"):
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
        device=device,
    )
    torch.jit.script(lstm)


def test_layernorm_lstm_with_projection_jit(device="cpu"):
    input_size = 2
    hidden_size = 5
    proj_size = 3
    num_layers = 4
    bias = True

    lstm = LayerNormLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        proj_size=proj_size,
        ln=nn.Identity,
        device=device,
    )
    torch.jit.script(lstm)


def test_layernorm_lstm_forward(device="cpu"):
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
        device=device,
    )
    torch_lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=True,
        bidirectional=False,
    ).to(device)
    assert len(self_lstm.state_dict()) == len(torch_lstm.state_dict())
    with torch.no_grad():
        for name, param in self_lstm.named_parameters():
            # name has the form layers.0.cell.weight_hh
            parts = name.split(".")
            layer_num = parts[1]
            getattr(torch_lstm, f"{parts[-1]}_l{layer_num}").copy_(param)

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size, device=device).requires_grad_()
    hs = [torch.rand(N, hidden_size, device=device) for _ in range(num_layers)]
    cs = [torch.rand(N, hidden_size, device=device) for _ in range(num_layers)]
    states = list(zip(hs, cs))

    x_clone = x.detach().clone().requires_grad_()

    self_y, self_states = self_lstm(x, states)

    h = torch.stack(hs)
    c = torch.stack(cs)
    torch_y, (torch_h, torch_c) = torch_lstm(x_clone, (h, c))

    assert_allclose(self_y, torch_y)

    self_h = torch.stack([s[0] for s in self_states])
    self_c = torch.stack([s[1] for s in self_states])

    assert_allclose(self_h, torch_h)
    assert_allclose(self_c, torch_c)

    s = self_y.reshape(-1)
    t = torch_y.reshape(-1)

    s_sum = (s * torch.arange(s.numel(), device=device)).sum()
    t_sum = (t * torch.arange(t.numel(), device=device)).sum()
    shc_sum = s_sum + self_h.sum() + self_c.sum()
    thc_sum = t_sum + torch_h.sum() + torch_c.sum()

    shc_sum.backward()
    thc_sum.backward()
    assert_allclose(x.grad, x_clone.grad)


def test_layernorm_lstm_with_projection_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=10, high=100, size=(1,)).item()
    proj_size = torch.randint(low=2, high=hidden_size, size=(1,)).item()
    num_layers = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0

    self_lstm = LayerNormLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        proj_size=proj_size,
        ln=nn.Identity,
        device=device,
    )
    torch_lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        proj_size=proj_size,
        batch_first=True,
        bidirectional=False,
    ).to(device)
    assert len(self_lstm.state_dict()) == len(torch_lstm.state_dict())
    with torch.no_grad():
        for name, param in self_lstm.named_parameters():
            # name has the form layers.0.cell.weight_hh
            parts = name.split(".")
            layer_num = parts[1]
            getattr(torch_lstm, f"{parts[-1]}_l{layer_num}").copy_(param)

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size, device=device).requires_grad_()
    hs = [torch.rand(N, proj_size, device=device) for _ in range(num_layers)]
    cs = [torch.rand(N, hidden_size, device=device) for _ in range(num_layers)]
    states = list(zip(hs, cs))

    x_clone = x.detach().clone().requires_grad_()

    self_y, self_states = self_lstm(x, states)

    h = torch.stack(hs)
    c = torch.stack(cs)
    torch_y, (torch_h, torch_c) = torch_lstm(x_clone, (h, c))

    assert_allclose(self_y, torch_y)

    self_h = torch.stack([s[0] for s in self_states])
    self_c = torch.stack([s[1] for s in self_states])

    assert_allclose(self_h, torch_h)
    assert_allclose(self_c, torch_c)

    s = self_y.reshape(-1)
    t = torch_y.reshape(-1)

    s_sum = (s * torch.arange(s.numel(), device=device)).sum()
    t_sum = (t * torch.arange(t.numel(), device=device)).sum()
    shc_sum = s_sum + self_h.sum() + self_c.sum()
    thc_sum = t_sum + torch_h.sum() + torch_c.sum()

    shc_sum.backward()
    thc_sum.backward()
    assert_allclose(x.grad, x_clone.grad)


def test_layernorm_gru_cell_jit(device="cpu"):
    input_size = 10
    hidden_size = 20
    cell = LayerNormGRUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        device=device,
    )

    torch.jit.script(cell)


def test_layernorm_gru_cell_constructor(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()

    self_cell = LayerNormGRUCell(
        input_size,
        hidden_size,
        ln=nn.Identity,
        device=device,
    )
    torch_cell = nn.GRUCell(
        input_size,
        hidden_size,
    ).to(device)

    for name, param in self_cell.named_parameters():
        assert param.shape == getattr(torch_cell, name).shape

    assert len(self_cell.state_dict()) == len(torch_cell.state_dict())


def test_layernorm_gru_cell_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0

    self_cell = LayerNormGRUCell(
        input_size,
        hidden_size,
        bias=bias,
        ln=nn.Identity,
        device=device,
    )
    torch_cell = nn.GRUCell(
        input_size,
        hidden_size,
        bias=bias,
    ).to(device)
    with torch.no_grad():
        for name, torch_param in torch_cell.named_parameters():
            self_param = getattr(self_cell, name)
            torch_param.copy_(self_param)

    N = torch.randint(low=2, high=100, size=(1,))
    x = torch.rand(N, input_size, device=device).requires_grad_()
    h = torch.rand(N, hidden_size, device=device)

    x_clone = x.detach().clone().requires_grad_()

    self_h = self_cell(x.clone(), h)
    torch_h = torch_cell(x_clone, h)

    assert_allclose(self_h, torch_h, atol=1e-5)

    (
        self_h.reshape(-1) * torch.arange(self_h.numel(), device=device)
    ).sum().backward()
    (
        torch_h.reshape(-1) * torch.arange(torch_h.numel(), device=device)
    ).sum().backward()

    assert_allclose(x.grad, x_clone.grad, atol=1e-3)


def test_layernorm_gru_layer_jit(device="cpu"):
    input_size = 10
    hidden_size = 20
    layer = LayerNormGRULayer(
        input_size,
        hidden_size=hidden_size,
        device=device,
    )
    torch.jit.script(layer)


def test_layernorm_gru_layer_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0
    self_layer = LayerNormGRULayer(
        input_size,
        hidden_size,
        bias=bias,
        ln=nn.Identity,
        device=device,
    )

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size, device=device).requires_grad_()
    h = torch.rand(N, hidden_size, device=device)

    x_clone = x.detach().clone().requires_grad_()

    self_y, self_h = self_layer(x, h.clone())

    torch_layer = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=bias,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    ).to(device)
    with torch.no_grad():
        for name, self_param in self_layer.cell.named_parameters():
            getattr(torch_layer, f"{name}_l0").copy_(self_param)

    torch_y, torch_h = torch_layer(x_clone, h.unsqueeze(0))
    assert_allclose(self_y, torch_y)
    assert_allclose(self_h, torch_h)

    self_y_sum = (
        self_y.reshape(-1) * torch.arange(self_y.numel(), device=device)
    ).sum()
    torch_y_sum = (
        torch_y.reshape(-1) * torch.arange(torch_y.numel(), device=device)
    ).sum()

    self_y_sum.backward()
    torch_y_sum.backward()

    assert_allclose(x.grad, x_clone.grad, atol=0.1)


def test_layernorm_gru_jit(device="cpu"):
    input_size = 2
    hidden_size = 3
    num_layers = 4
    bias = True

    gru = LayerNormGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        ln=nn.Identity,
        device=device,
    )
    torch.jit.script(gru)


def test_layernorm_gru_forward(device="cpu"):
    input_size = torch.randint(low=2, high=100, size=(1,)).item()
    hidden_size = torch.randint(low=2, high=100, size=(1,)).item()
    num_layers = torch.randint(low=2, high=100, size=(1,)).item()
    bias = torch.randint(low=0, high=1000, size=(1,)).item() & 2 == 0

    self_gru = LayerNormGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        ln=nn.Identity,
        device=device,
    )
    torch_gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=True,
        bidirectional=False,
    ).to(device)
    assert len(self_gru.state_dict()) == len(torch_gru.state_dict())
    with torch.no_grad():
        for name, param in self_gru.named_parameters():
            # name has the form layers.0.cell.weight_hh
            parts = name.split(".")
            layer_num = parts[1]
            getattr(torch_gru, f"{parts[-1]}_l{layer_num}").copy_(param)

    N = torch.randint(low=2, high=100, size=(1,))
    T = torch.randint(low=2, high=100, size=(1,))

    x = torch.rand(N, T, input_size, device=device).requires_grad_()
    states = [
        torch.rand(N, hidden_size, device=device) for _ in range(num_layers)
    ]

    x_clone = x.detach().clone().requires_grad_()

    self_y, self_states = self_gru(x, states)

    torch_y, torch_states = torch_gru(x_clone, torch.stack(states))

    assert_allclose(self_y, torch_y)

    self_states = torch.stack(self_states)

    assert_allclose(self_states, torch_states)

    s = self_y.reshape(-1)
    t = torch_y.reshape(-1)

    s_sum = (s * torch.arange(s.numel(), device=device)).sum()
    t_sum = (t * torch.arange(t.numel(), device=device)).sum()
    s_state_sum = s_sum + self_states.sum()
    t_state_sum = t_sum + torch_states.sum()

    s_state_sum.backward()
    t_state_sum.backward()
    assert_allclose(x.grad, x_clone.grad, atol=1e-2)


def _test_lstm(device):
    test_layernorm_lstm_cell_jit(device)
    test_layernorm_lstm_cell_constructor(device)
    test_layernorm_lstm_cell_with_projection_jit(device)
    test_layernorm_lstm_cell_forward(device)
    test_layernorm_lstm_cell_with_projection_forward(device)
    #
    test_layernorm_lstm_layer_jit(device)
    test_layernorm_lstm_layer_with_project_jit(device)
    test_layernorm_lstm_layer_forward(device)
    test_layernorm_lstm_layer_with_projection_forward(device)

    test_layernorm_lstm_jit(device)
    test_layernorm_lstm_with_projection_jit(device)
    test_layernorm_lstm_forward(device)
    test_layernorm_lstm_with_projection_forward(device)


def _test_gru(device):
    test_layernorm_gru_cell_jit(device)
    test_layernorm_gru_cell_constructor(device)
    test_layernorm_gru_cell_forward(device)
    #
    test_layernorm_gru_layer_jit(device)
    test_layernorm_gru_layer_forward(device)
    #
    test_layernorm_gru_jit(device)
    test_layernorm_gru_forward(device)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def main():
    for device in get_devices():
        print("device", device)
        _test_lstm(device)
        _test_gru(device)


if __name__ == "__main__":
    torch.manual_seed(20211202)
    main()
