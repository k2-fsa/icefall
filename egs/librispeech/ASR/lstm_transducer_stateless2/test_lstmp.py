#!/usr/bin/env python3

import torch
import torch.nn as nn
from lstmp import LSTMP


def test():
    input_size = torch.randint(low=10, high=1024, size=(1,)).item()
    hidden_size = torch.randint(low=10, high=1024, size=(1,)).item()
    proj_size = hidden_size - 1
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        proj_size=proj_size,
    )
    lstmp = LSTMP(lstm)

    N = torch.randint(low=1, high=10, size=(1,)).item()
    T = torch.randint(low=1, high=20, size=(1,)).item()
    x = torch.rand(T, N, input_size)
    h0 = torch.rand(1, N, proj_size)
    c0 = torch.rand(1, N, hidden_size)

    y1, (h1, c1) = lstm(x, (h0, c0))
    y2, (h2, c2) = lstmp(x, (h0, c0))

    assert torch.allclose(y1, y2, atol=1e-5), (y1 - y2).abs().max()
    assert torch.allclose(h1, h2, atol=1e-5), (h1 - h2).abs().max()
    assert torch.allclose(c1, c2, atol=1e-5), (c1 - c2).abs().max()

    #  lstm_script = torch.jit.script(lstm) # pytorch does not support it
    lstm_script = lstm
    lstmp_script = torch.jit.script(lstmp)

    y3, (h3, c3) = lstm_script(x, (h0, c0))
    y4, (h4, c4) = lstmp_script(x, (h0, c0))

    assert torch.allclose(y3, y4, atol=1e-5), (y3 - y4).abs().max()
    assert torch.allclose(h3, h4, atol=1e-5), (h3 - h4).abs().max()
    assert torch.allclose(c3, c4, atol=1e-5), (c3 - c4).abs().max()

    assert torch.allclose(y3, y1, atol=1e-5), (y3 - y1).abs().max()
    assert torch.allclose(h3, h1, atol=1e-5), (h3 - h1).abs().max()
    assert torch.allclose(c3, c1, atol=1e-5), (c3 - c1).abs().max()

    lstm_trace = torch.jit.trace(lstm, (x, (h0, c0)))
    lstmp_trace = torch.jit.trace(lstmp, (x, (h0, c0)))

    y5, (h5, c5) = lstm_trace(x, (h0, c0))
    y6, (h6, c6) = lstmp_trace(x, (h0, c0))

    assert torch.allclose(y5, y6, atol=1e-5), (y5 - y6).abs().max()
    assert torch.allclose(h5, h6, atol=1e-5), (h5 - h6).abs().max()
    assert torch.allclose(c5, c6, atol=1e-5), (c5 - c6).abs().max()

    assert torch.allclose(y5, y1, atol=1e-5), (y5 - y1).abs().max()
    assert torch.allclose(h5, h1, atol=1e-5), (h5 - h1).abs().max()
    assert torch.allclose(c5, c1, atol=1e-5), (c5 - c1).abs().max()


@torch.no_grad()
def main():
    test()


if __name__ == "__main__":
    main()
