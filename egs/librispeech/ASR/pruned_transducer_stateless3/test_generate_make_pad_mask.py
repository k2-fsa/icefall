#!/usr/bin/env python3


import torch
import torch.nn as nn
from conformer import MakePadMask


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.make_pad_mask = MakePadMask()

    def forward(self, x: torch.Tensor):
        """
        Args:
          x:
            (N,)
        """
        src_key_padding_mask = self.make_pad_mask(x)
        return src_key_padding_mask


def generate_pt():
    f = Foo()
    f.eval()
    x = torch.tensor([1, 3, 5])
    y = f(x)
    print("y.shape", y.shape)
    print(y)
    m = torch.jit.trace(f, x)
    m.save("foo/make_pad_mask.pt")
    print(m.graph)


def main():
    generate_pt()


if __name__ == "__main__":
    torch.manual_seed(20220809)
    main()
