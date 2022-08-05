#!/usr/bin/env python3

import math

import ncnn
import numpy as np
import torch
import torch.nn as nn
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv2d,
    ScaledLinear,
)

LOG_EPS = math.log(1e-10)


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        layer1_channels = 8
        layer2_channels = 32
        layer3_channels = 128
        in_channels = 80
        out_channels = 512
        self.out_channels = out_channels
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
        print(self.out.weight.shape)
        self.out_norm = BasicNorm(out_channels, eps=1, learn_eps=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)

        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        #  b, c, t, f = x.shape
        x = self.out(x.contiguous().view(1, -1, 128 * 19))

        x = self.out_norm(x)
        return x


@torch.no_grad()
def main():
    x = torch.rand(1, 200, 80)
    f = torch.jit.load("foo/scaled_conv2d.pt")

    param = "foo/scaled_conv2d.ncnn.param"
    model = "foo/scaled_conv2d.ncnn.bin"

    with ncnn.Net() as net:
        net.load_param(param)
        net.load_model(model)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy()).clone())
            ret, out0 = ex.extract("out0")
            assert ret == 0
            out0 = np.array(out0)
            print("ncnn", out0.shape)
            t = f(x)
            out0 = torch.from_numpy(out0)
            t = t.squeeze(0)
            print("torch", t.shape)
            torch.allclose(out0, t), (t - out0).abs().max()


if __name__ == "__main__":
    main()
