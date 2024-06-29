#!/usr/bin/env python3

from zipformer import ContextualPositionalEncoding


def test():
    embed_dim = 5
    npos_max = 10
    cope = ContextualPositionalEncoding(embed_dim=embed_dim, npos_max=npos_max)
    q = torch.rand(2, 3, 4, embed_dim)
    qk = torch.rand(2, 3, 4, 6)

    p = cope(q=q, qk=qk)
    print(p.shape)


def main():
    test()


if __name__ == "__main__":
    main()
