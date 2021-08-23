#!/usr/bin/env python3
# run with:
#  python3 -m pytest test_conformer.py

import torch
from conformer import (
    TransformerDecoderRelPos,
    MaskedLmConformer,
    RelPositionMultiheadAttention,
    RelPositionalEncoding,
    generate_square_subsequent_mask,
)

from torch.nn.utils.rnn import pad_sequence


def test_rel_position_multihead_attention():
    # Also tests RelPositionalEncoding
    embed_dim = 256
    num_heads = 4
    T = 25
    N = 4
    C = 256
    pos_emb_module = RelPositionalEncoding(C, dropout_rate=0.0)
    rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)

    x = torch.randn(N, T, C)
    #pos_emb = torch.randn(1, 2*T-1, C)
    x, pos_enc = pos_emb_module(x)
    print("pos_enc.shape=", pos_enc.shape)
    x = x.transpose(0, 1)  # (T, N, C)
    attn_output, attn_output_weights = rel_pos_multihead_attn(x, x, x, pos_enc)


def test_transformer():
    return
    num_features = 40
    num_classes = 87
    model = Transformer(num_features=num_features, num_classes=num_classes)

    N = 31

    for T in range(7, 30):
        x = torch.rand(N, T, num_features)
        y, _, _ = model(x)
        assert y.shape == (N, (((T - 1) // 2) - 1) // 2, num_classes)


def test_generate_square_subsequent_mask():
    s = 5
    mask = generate_square_subsequent_mask(s, torch.device('cpu'))
    inf = float("inf")
    expected_mask = torch.tensor(
        [
            [0.0, -inf, -inf, -inf, -inf],
            [0.0, 0.0, -inf, -inf, -inf],
            [0.0, 0.0, 0.0, -inf, -inf],
            [0.0, 0.0, 0.0, 0.0, -inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.all(torch.eq(mask, expected_mask))
