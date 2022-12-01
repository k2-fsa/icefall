# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

import torch
import torch.nn.functional as F

from icefall.utils import add_eos, add_sos, make_pad_mask


class RnnLmModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        tie_weights: bool = False,
    ):
        """
        Args:
          vocab_size:
            Vocabulary size of BPE model.
          embedding_dim:
            Input embedding dimension.
          hidden_dim:
            Hidden dimension of RNN layers.
          num_layers:
            Number of RNN layers.
          tie_weights:
            True to share the weights between the input embedding layer and the
            last output linear layer. See https://arxiv.org/abs/1608.05859
            and https://arxiv.org/abs/1611.01462
        """
        super().__init__()

        self.input_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.rnn = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_linear = torch.nn.Linear(
            in_features=hidden_dim, out_features=vocab_size
        )

        self.vocab_size = vocab_size
        if tie_weights:
            logging.info("Tying weights")
            assert embedding_dim == hidden_dim, (embedding_dim, hidden_dim)
            self.output_linear.weight = self.input_embedding.weight
        else:
            logging.info("Not tying weights")

        self.cache = {}

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor with shape (N, L). Each row
            contains token IDs for a sentence and starts with the SOS token.
          y:
            A shifted version of `x` and with EOS appended.
          lengths:
            A 1-D tensor of shape (N,). It contains the sentence lengths
            before padding.
        Returns:
          Return a 2-D tensor of shape (N, L) containing negative log-likelihood
          loss values. Note: Loss values for padding positions are set to 0.
        """
        assert x.ndim == y.ndim == 2, (x.ndim, y.ndim)
        assert lengths.ndim == 1, lengths.ndim
        assert x.shape == y.shape, (x.shape, y.shape)

        batch_size = x.size(0)
        assert lengths.size(0) == batch_size, (lengths.size(0), batch_size)

        # embedding is of shape (N, L, embedding_dim)
        embedding = self.input_embedding(x)

        # Note: We use batch_first==True
        rnn_out, _ = self.rnn(embedding)
        logits = self.output_linear(rnn_out)

        # Note: No need to use `log_softmax()` here
        # since F.cross_entropy() expects unnormalized probabilities

        # nll_loss is of shape (N*L,)
        # nll -> negative log-likelihood
        nll_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), y.reshape(-1), reduction="none"
        )
        # Set loss values for padding positions to 0
        mask = make_pad_mask(lengths).reshape(-1)
        nll_loss.masked_fill_(mask, 0)

        nll_loss = nll_loss.reshape(batch_size, -1)

        return nll_loss

    def predict_batch(self, tokens, token_lens, sos_id, eos_id, blank_id):
        device = next(self.parameters()).device
        batch_size = len(token_lens)

        sos_tokens = add_sos(tokens, sos_id)
        tokens_eos = add_eos(tokens, eos_id)
        sos_tokens_row_splits = sos_tokens.shape.row_splits(1)

        sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]

        x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
        y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)

        x_tokens = x_tokens.to(torch.int64).to(device)
        y_tokens = y_tokens.to(torch.int64).to(device)
        sentence_lengths = sentence_lengths.to(torch.int64).to(device)

        embedding = self.input_embedding(x_tokens)

        # Note: We use batch_first==True
        rnn_out, states = self.rnn(embedding)
        logits = self.output_linear(rnn_out)
        mask = torch.zeros(logits.shape).bool().to(device)
        for i in range(batch_size):
            mask[i, token_lens[i], :] = True
        logits = logits[mask].reshape(batch_size, -1)

        return logits[:, :].log_softmax(-1), states

    def clean_cache(self):
        self.cache = {}

    def score_token(self, tokens: torch.Tensor, state=None):
        device = next(self.parameters()).device
        batch_size = tokens.size(0)
        if state:
            h, c = state
        else:
            h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.input_size).to(
                device
            )
            c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.input_size).to(
                device
            )

        embedding = self.input_embedding(tokens)
        rnn_out, states = self.rnn(embedding, (h, c))
        logits = self.output_linear(rnn_out)

        return logits[:, 0].log_softmax(-1), states

    def forward_with_state(
        self, tokens, token_lens, sos_id, eos_id, blank_id, state=None
    ):
        batch_size = len(token_lens)
        if state:
            h, c = state
        else:
            h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.input_size)
            c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.input_size)

        device = next(self.parameters()).device

        sos_tokens = add_sos(tokens, sos_id)
        tokens_eos = add_eos(tokens, eos_id)
        sos_tokens_row_splits = sos_tokens.shape.row_splits(1)

        sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]

        x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
        y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)

        x_tokens = x_tokens.to(torch.int64).to(device)
        y_tokens = y_tokens.to(torch.int64).to(device)
        sentence_lengths = sentence_lengths.to(torch.int64).to(device)

        embedding = self.input_embedding(x_tokens)

        # Note: We use batch_first==True
        rnn_out, states = self.rnn(embedding, (h, c))
        logits = self.output_linear(rnn_out)

        return logits, states
