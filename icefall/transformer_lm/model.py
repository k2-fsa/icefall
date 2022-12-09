# Copyright (c)  2021  Xiaomi Corporation (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from encoder import Transformer
from mask import subsequent_mask

from icefall.utils import AttributeDict, add_eos, add_sos, make_pad_mask


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        tie_weights: bool = True,
        dropout: float = 0.1,
        emb_dropout_rate: float = 0.0,
        params: AttributeDict = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.params = params

        self.input_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.encoder = Transformer(
            input_dim=embedding_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            dropout_rate=dropout,
        )

        self.output_linear = torch.nn.Linear(
            in_features=d_model, out_features=vocab_size
        )
        if tie_weights:
            logging.info("Tying weights")
            assert d_model == embedding_dim, (d_model, embedding_dim)
            self.output_linear.weight = self.input_embedding.weight
        else:
            logging.info("Not tying weights")

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_lens: torch.Tensor):

        x = self.input_embedding(x)

        x, x_lens = self.encoder(x, x_lens)

        logits = self.output_linear(x)

        nll_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), y.reshape(-1), reduction="none"
        )

        mask = make_pad_mask(x_lens).reshape(-1)
        nll_loss.masked_fill_(mask, 0)

        return nll_loss
