#!/usr/bin/env python3


import torch
import torch.nn as nn
from conformer import Conv2dSubsampling
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(self):
        super().__init__()

        num_features = 80
        subsampling_factor = 4
        d_model = 512
        self.num_features = num_features
        self.subsampling_factor = subsampling_factor

        self.encoder_embed = Conv2dSubsampling(
            num_features,
            d_model,
            for_pnnx=True,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
          x:
            (N, T, C)
        """
        x = self.encoder_embed(x)
        return x


def generate_pt():
    f = Foo()
    f.eval()
    f = convert_scaled_to_non_scaled(f)
    x = torch.rand(1, 30, 80)
    y = f(x)
    print("y.shape", y.shape)
    m = torch.jit.trace(f, x)
    m.save("foo/encoder_embed.pt")


def main():
    generate_pt()


if __name__ == "__main__":
    torch.manual_seed(20220809)
    main()
