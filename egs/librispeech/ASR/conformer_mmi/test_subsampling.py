#!/usr/bin/env python3

import torch
from subsampling import Conv2dSubsampling, VggSubsampling


def test_conv2d_subsampling():
    N = 3
    odim = 2

    for T in range(7, 19):
        for idim in range(7, 20):
            model = Conv2dSubsampling(idim=idim, odim=odim)
            x = torch.empty(N, T, idim)
            y = model(x)
            assert y.shape[0] == N
            assert y.shape[1] == ((T - 1) // 2 - 1) // 2
            assert y.shape[2] == odim


def test_vgg_subsampling():
    N = 3
    odim = 2

    for T in range(7, 19):
        for idim in range(7, 20):
            model = VggSubsampling(idim=idim, odim=odim)
            x = torch.empty(N, T, idim)
            y = model(x)
            assert y.shape[0] == N
            assert y.shape[1] == ((T - 1) // 2 - 1) // 2
            assert y.shape[2] == odim
