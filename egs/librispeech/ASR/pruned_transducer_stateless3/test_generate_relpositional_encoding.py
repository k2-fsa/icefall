#!/usr/bin/env python3


import torch
import torch.nn as nn
from conformer import RelPositionalEncoding
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 512
        dropout = 0.1

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
          x:
            (N, T, C)
        """
        y, pos_emb = self.encoder_pos(x)
        return y, pos_emb


def generate_pt():
    f = Foo()
    f.eval()
    f = convert_scaled_to_non_scaled(f)
    x = torch.rand(1, 6, 4)
    y, pos_emb = f(x)
    print("y.shape", y.shape)
    print("pos_emb.shape", pos_emb.shape)
    m = torch.jit.trace(f, x)
    m.save("foo/encoder_pos.pt")
    print(m.encoder_pos.pe[0].shape)
    print(type(m.encoder_pos.pe[0]))
    print(m.graph)


def main():
    generate_pt()


if __name__ == "__main__":
    torch.manual_seed(20220809)
    main()
