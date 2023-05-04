#!/usr/bin/env python3
# Copyright    2023       Xiaomi Corp.        (authors: Daniel Povey)
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
import random
import torch
from torch import nn, Tensor


class ChunkDecoder(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim: int,
                 chunk_size: int,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int = 2):
        """
        A 'decoder' that computes the probability of symbols in a language modeling task.
        Conceptually it computes the probability of `chunk_size` symbols (e.g. 8 symbols)
        based on an embedding derived from all symbols preceding this chunk of 8 symbols.
        Also, within the chunk, we always see all previous symbols (plus the last symbol
        of the previous chunk).
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers)

        self.label_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim)

        # project to hidden state and cell state at the beginning of each chunk.
        # (we don't run the lstm contiuously over the sequence, for both
        # training speed and stability; instead, we reconstruct the hidden
        # state for each chunk.)
        # the factor of 2 is to cover hidden state and cell state.
        self.init_proj = nn.Linear(embed_dim,
                                   2 * hidden_size * num_layers)

        self.out_proj = nn.Linear(hidden_size,
                                  vocab_size)


    def forward(self,
                labels: Tensor,
                encoder_embed: Tensor) -> Tensor:
        """
        Compute log-probs.
        Args:
           labels: the labels, a Tensor of integer type of shape (batch_size, seq_len);
                 seq_len is expected to be a multiple of chunk_size.
        encoder_embed: the embeddings from the encoder, of shape (seq_len//chunk_size, batch_size, embed_dim)

        Returns:
            returns the log-probs for each symbol, in a Tensor of shape (batch_size, seq_len).
        """
        (batch_size, seq_len) = labels.shape
        (num_chunks, _batch_size, embed_dim) = encoder_embed.shape
        chunk_size = self.chunk_size
        assert batch_size == _batch_size
        assert num_chunks * chunk_size == seq_len

        labels_shifted = torch.cat((torch.zeros_like(labels[0:1]),
                                    labels[:-1]), dim=0)

        labels_embed = self.label_embed(labels_shifted.t())  # (seq_len, batch_size, embed_dim)

        init = self.init_proj(encoder_embed).reshape(num_chunks, batch_size,
                                                     2, self.num_layers, self.hidden_size)
        init = init.permute(2, 3, 0, 1, 4).reshape(2, self.num_layers,
                                                   num_chunks * batch_size,
                                                   self.hidden_size).contiguous()
        hidden = init[0]
        cell = init[1]


        labels_embed = labels_embed.reshape(num_chunks, chunk_size, batch_size, embed_dim).transpose(0, 1)
        labels_embed = labels_embed.contiguous().reshape(chunk_size, num_chunks * batch_size, embed_dim)
        encoder_embed = encoder_embed.reshape(1, num_chunks * batch_size, embed_dim)

        x = labels_embed + encoder_embed  # broadcasts encoder_embed over the chunk_size

        (x, _hidden) = self.lstm(x, hx=(hidden, cell))

        x = self.out_proj(x)

        vocab_size = x.shape[-1]
        # x: (chunk_size, num_chunks * batch_size, vocab_size)
        x = x.reshape(chunk_size, num_chunks, batch_size, vocab_size)
        x = x.permute(2, 1, 0, 3).contiguous().reshape(batch_size, num_chunks * chunk_size, vocab_size)

        x = x.log_softmax(dim=-1)

        logprobs = torch.gather(x, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

        if random.random() < 0.02:
            # occasionally print out average logprob per position in the chunk.
            l = logprobs.reshape(batch_size, num_chunks, chunk_size).mean(dim=(0, 1))
            l = l.to('cpu').tolist()
            logging.info(f"Logprobs per position in chunk: {l}")

        return logprobs
