#!/usr/bin/env python3


import torch
import torch.nn as nn
from conformer import Conv2dSubsampling, MakePadMask, RelPositionalEncoding
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        num_features = 80
        d_model = 512
        dropout = 0.1

        self.num_features = num_features
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)
        self.encoder_pos = RelPositionalEncoding(d_model, dropout)
        self.make_pad_mask = MakePadMask()

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """
        Args:
          x:
            (N,T,C)
          x_lens:
            (N,)
        """
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        lengths1 = torch.floor((x_lens - 1) / 2)
        lengths = torch.floor((lengths1 - 1) / 2)
        lengths = lengths.to(x_lens)

        return x, lengths, pos_emb


def generate_pt():
    f = Foo()
    f.eval()
    f = convert_scaled_to_non_scaled(f)
    f.encoder_embed.for_pnnx = True

    x = torch.rand(1, 30, 80, dtype=torch.float32)  # (T, C)
    x_lens = torch.tensor([30])
    y, lengths, pos_emb = f(x, x_lens)
    print("y.shape", y.shape)
    print("lengths", lengths)
    print("pos_emb.shape", pos_emb.shape)
    m = torch.jit.trace(f, (x, x_lens))
    m.save("foo/conformer.pt")
    #  print(m.graph)


def main():
    generate_pt()


if __name__ == "__main__":
    torch.manual_seed(20220809)
    main()
