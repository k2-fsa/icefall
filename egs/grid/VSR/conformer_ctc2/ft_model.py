#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""AV-HuBERT visual encoder, truncated to the --layer layers whose output the
frozen recipe taps, fine-tuned end-to-end with the Conformer-CTC head.
Trained by train_ft.py and decoded by decode.py --finetuned true.
"""
import torch
import torch.nn as nn
from conformer import Conformer


class FTConformer(nn.Module):
    """Trainable AV-HuBERT encoder + Conformer-CTC head.

    Input is (N, 1, T, H, W) normalized ROI frames; forward mirrors
    Conformer's interface.
    """

    def __init__(
        self,
        avhubert: nn.Module,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        subsampling_factor: int = 1,
        dropout: float = 0.1,
        layer_dropout: float = 0.1,
    ):
        super().__init__()
        self.avhubert = avhubert
        self.conformer = Conformer(
            num_features=avhubert.encoder.embedding_dim,  # 768 Base, 1024 Large
            num_classes=num_classes,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

    def forward(self, x: torch.Tensor, supervisions=None, warmup: float = 1.0):
        # The frozen recipe extracts with output_layer=--layer. With
        # layer_norm_first, fairseq applies the final encoder.layer_norm only
        # when output_layer is None, so pass the layer count to tap the same
        # representation.
        n_layers = len(self.avhubert.encoder.layers)
        feats, _ = self.avhubert.extract_finetune(
            source={"video": x, "audio": None},
            padding_mask=None,
            output_layer=n_layers,
        )
        return self.conformer(feats, supervisions, warmup=warmup)

    def decoder_forward(self, *args, **kwargs):
        return self.conformer.decoder_forward(*args, **kwargs)
