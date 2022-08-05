#!/usr/bin/env python3


import torch
import torch.nn as nn
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv2d,
    ScaledLinear,
)
from scaling_converter import convert_scaled_to_non_scaled


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
        self.out_norm = BasicNorm(out_channels, learn_eps=False)

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


def generate_scaled_conv2d():
    print("generating")
    f = Foo()
    f.eval()
    f = convert_scaled_to_non_scaled(f)
    print(f)
    torch.save(f.state_dict(), "f.pt")
    x = torch.rand(1, 100, 80)  # NTC
    m = torch.jit.trace(f, x)
    m.save("foo/scaled_conv2d.pt")
    print(m.graph)


@torch.no_grad()
def main():
    generate_scaled_conv2d()


if __name__ == "__main__":
    torch.manual_seed(20220803)
    main()
