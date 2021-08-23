#!/usr/bin/env python3
# run with:
#  python3 -m pytest test_conformer.py

import torch
from conformer import (
    TransformerDecoderRelPos,
    TransformerDecoderLayerRelPos,
    MaskedLmConformer,
    MaskedLmConformerEncoder,
    MaskedLmConformerEncoderLayer,
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
    x, pos_emb = pos_emb_module(x)
    x = x.transpose(0, 1)  # (T, N, C)
    attn_output, attn_output_weights = rel_pos_multihead_attn(x, x, x, pos_emb)


def test_masked_lm_conformer_encoder_layer():
    # Also tests RelPositionalEncoding
    embed_dim = 256
    num_heads = 4
    T = 25
    N = 4
    C = 256
    pos_emb_module = RelPositionalEncoding(C, dropout_rate=0.0)
    encoder_layer = MaskedLmConformerEncoderLayer(embed_dim, num_heads)


    x = torch.randn(N, T, C)
    x, pos_emb = pos_emb_module(x)
    x = x.transpose(0, 1)  # (T, N, C)
    key_padding_mask = (torch.randn(N, T) > 0.0)  # (N, T)
    y = encoder_layer(x, pos_emb, key_padding_mask=key_padding_mask)


def test_masked_lm_conformer_encoder():
    # Also tests RelPositionalEncoding
    embed_dim = 256
    num_heads = 4
    T = 25
    N = 4
    C = 256
    pos_emb_module = RelPositionalEncoding(C, dropout_rate=0.0)
    encoder_layer = MaskedLmConformerEncoderLayer(embed_dim, num_heads)
    norm = torch.nn.LayerNorm(embed_dim)
    encoder = MaskedLmConformerEncoder(encoder_layer, num_layers=4,
                                       norm=norm)


    x = torch.randn(N, T, C)
    x, pos_emb = pos_emb_module(x)
    x = x.transpose(0, 1)  # (T, N, C)
    key_padding_mask = (torch.randn(N, T) > 0.0)  # (N, T)
    y = encoder(x, pos_emb, key_padding_mask=key_padding_mask)


def test_transformer_decoder_layer_rel_pos():
    embed_dim = 256
    num_heads = 4
    T = 25
    N = 4
    C = 256
    pos_emb_module = RelPositionalEncoding(C, dropout_rate=0.0)
    decoder_layer = TransformerDecoderLayerRelPos(embed_dim, num_heads)


    x = torch.randn(N, T, C)
    x, pos_emb = pos_emb_module(x)
    x = x.transpose(0, 1)  # (T, N, C)
    key_padding_mask = (torch.randn(N, T) > 0.0)  # (N, T)
    attn_mask = generate_square_subsequent_mask(T)
    memory = torch.randn(T, N, C)
    y = decoder_layer(x, pos_emb, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask)



def test_transformer_decoder_rel_pos():
    embed_dim = 256
    num_heads = 4
    T = 25
    N = 4
    C = 256
    pos_emb_module = RelPositionalEncoding(C, dropout_rate=0.0)
    decoder_layer = TransformerDecoderLayerRelPos(embed_dim, num_heads)
    decoder_norm = torch.nn.LayerNorm(embed_dim)
    decoder = TransformerDecoderRelPos(decoder_layer, num_layers=6, norm=decoder_norm)


    x = torch.randn(N, T, C)
    x, pos_emb = pos_emb_module(x)
    x = x.transpose(0, 1)  # (T, N, C)
    key_padding_mask = (torch.randn(N, T) > 0.0)  # (N, T)
    attn_mask = generate_square_subsequent_mask(T)
    memory = torch.randn(T, N, C)
    y = decoder(x, pos_emb, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


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
