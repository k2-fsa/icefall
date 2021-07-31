#!/usr/bin/env python3

import pytest
import torch
import torch.nn as nn

from icefall.checkpoint import (
    average_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def checkpoints1(tmp_path):
    f = tmp_path / "f.pt"
    m = nn.Module()
    m.p1 = nn.Parameter(torch.tensor([10.0, 20.0]), requires_grad=False)
    m.register_buffer("p2", torch.tensor([10, 100]))

    params = {"a": 10, "b": 20}
    save_checkpoint(f, m, params=params)
    return f


@pytest.fixture
def checkpoints2(tmp_path):
    f = tmp_path / "f2.pt"
    m = nn.Module()
    m.p1 = nn.Parameter(torch.Tensor([50, 30.0]))
    m.register_buffer("p2", torch.tensor([1, 3]))
    params = {"a": 100, "b": 200}

    save_checkpoint(f, m, params=params)
    return f


def test_load_checkpoints(checkpoints1):
    m = nn.Module()
    m.p1 = nn.Parameter(torch.Tensor([0, 0.0]))
    m.p2 = nn.Parameter(torch.Tensor([0, 0]))
    params = load_checkpoint(checkpoints1, m)
    assert torch.allclose(m.p1, torch.Tensor([10.0, 20]))
    assert params["a"] == 10
    assert params["b"] == 20


def test_average_checkpoints(checkpoints1, checkpoints2):
    state_dict = average_checkpoints([checkpoints1, checkpoints2])
    assert torch.allclose(state_dict["p1"], torch.Tensor([30, 25.0]))
    assert torch.allclose(state_dict["p2"], torch.tensor([5, 51]))
