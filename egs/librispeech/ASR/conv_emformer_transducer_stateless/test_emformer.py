import torch


def test_rel_positional_encoding():
    from emformer import RelPositionalEncoding

    D = 256
    pos_enc = RelPositionalEncoding(D, dropout=0.1)
    pos_len = 100
    neg_len = 100
    x = torch.randn(2, D)
    x, pos_emb = pos_enc(x, pos_len, neg_len)
    assert pos_emb.shape == (pos_len + neg_len - 1, D)


if __name__ == "__main__":
    test_rel_positional_encoding()
