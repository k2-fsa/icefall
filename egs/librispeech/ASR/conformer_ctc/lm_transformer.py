#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

from typing import Any
from typing import List
from typing import Tuple

import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import generate_square_subsequent_mask
from transformer import make_pad_mask
from transformer import TransformerEncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        embed_unit=128,
        d_model=512,
        nhead=8,
        attention_dropout_rate=0.0,
        num_encoder_layers=16,
        dim_feedforward=2048,
        normalize_before=True,
    ):
        super().__init__()

        self.input_embed = nn.Sequential(
            nn.Linear(embed_unit, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(attention_dropout_rate),
            nn.ReLU(),
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=attention_dropout_rate,
        )

        self.encoders = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, nn.LayerNorm(d_model)
        )

    def forward(self, xs, token_lens):
        # xs: N S E
        xs = self.input_embed(xs)
        mask = generate_square_subsequent_mask(xs.shape[1]).to(xs.device)

        src_key_padding_mask = make_pad_mask(token_lens).to(xs.device)

        # xs: N S E --> S N E
        xs = xs.transpose(0, 1)
        xs = self.encoders(
            xs, mask=mask, src_key_padding_mask=src_key_padding_mask
        )
        # xs: S N E --> N S E
        xs = xs.transpose(0, 1)

        return xs


class TransformerLM(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 16,
        vocab_size: int = 5000,
        embed_unit: int = 128,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.0,
        ignore_id: int = 0,
    ):
        super().__init__()

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id

        self.embed = nn.Embedding(vocab_size, embed_unit)

        self.encoder = Encoder(
            embed_unit=embed_unit,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
        )

        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        token_lens,
    ) -> Tuple[torch.Tensor, None]:
        # src: N, S
        x = self.embed(src)
        h = self.encoder(x, token_lens)
        # y: N, S, E
        y = self.decoder(h)
        return y

    def nll(self, xs_pad, target_pad, token_lens):
        # xs_pad/target_pad: N, S
        # An example element of xs_pad:
        # <sos> token token token ... token
        #
        # An example element of target_pad:
        # token token token ... token <eos>

        y = self.forward(xs_pad, token_lens)

        # nll: (N * S,)
        nll = F.cross_entropy(
            y.view(-1, y.shape[-1]), target_pad.view(-1), reduction="none"
        )

        # assign padded postion with 0.0
        nll.masked_fill_(make_pad_mask(token_lens).to(nll.device).view(-1), 0.0)

        # nll: (N * S,) -> (N, S)
        nll = nll.view(xs_pad.size(0), -1)
        return nll
