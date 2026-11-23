#!/usr/bin/env python3

import torch
from zipformer import ContextualPositionalEncoding


def test():
    embed_dim = 5
    npos_max = 10

    cope = ContextualPositionalEncoding(embed_dim=embed_dim, npos_max=npos_max)
    q = torch.rand(2, 3, npos_max, embed_dim)

    qk = torch.rand(2, 3, npos_max, npos_max)

    p = cope(q=q, qk=qk)
    print(p.shape)


def main():
    test()


if __name__ == "__main__":
    torch.manual_seed(20240703)
    main()
