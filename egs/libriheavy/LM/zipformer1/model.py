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



import torch
from torch import nn, Tensor
from subformer import Subformer
from scaling import Balancer


class TextEmbedder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int):
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim,
                               embedding_dim,
                               groups=embedding_dim,
                               kernel_size=2)
        self.balancer = Balancer(embedding_dim,
                                 channel_dim=-1,
                                 min_positive=0.1,
                                 min_abs=1.0,
                                 max_abs=2.0)
        self.activation1 = nn.ReLU()
        self.out_proj = nn.Linear(embedding_dim,
                                  embedding_dim,
                                  bias=False)

    def forward(self,
                text: Tensor) -> Tensor:
        """
        Args:
            text: Tensor of shape (seq_len, batch_size), containing integer indexes
                 0 <= text < vocab_size.
        Returns:
            Tensor of shape (seq_len, batch_size, embedding_dim)
        """
        x = self.embed(text)  # (seq_len, batch_size, embedding_dim)

        x = torch.cat((torch.zeros_like(x[0:1], x)), dim=0)  # pad
        x = x.permute(1, 2, 0)  # N,C,H, i.e. (batch_size, embedding_dim, seq_len)
        x = self.conv1(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, embedding_dim)
        x = self.balancer(x)  # make sure no channel has all zeros.
        x = self.activation1(x)
        x = self.out_proj(x)
        return x

class SubformerLM(nn.Module):

    def __init__(self,
                 encoder_embed: nn.Module,
                 encoder: Subformer,
                 decoder: nn.Module):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder # does subsampling
        self.decoder = decoder


    def forward(self,
                labels: Tensor):
        """
        Compute array of log-probs

        Args:
         labels: a Tensor containing the labels (in the range 0..num_symbols-1), of shape (batch_size, seq_len).
        Returns:
           a Tensor containing the log-probs for each label, of shape (batch_size, seq_len).
        """
        (batch_size, seq_len) = labels.shape

        chunk_size = 1
        labels_shifted = labels.t()  # (time, batch)
        labels_shifted = torch.cat((torch.zeros_like(labels_shifted[:1]),
                                    labels_shifted[:-1]),
                                   dim=0)

        x = self.encoder_embed(labels_shifted)
        x_lens = torch.full((batch_size,), seq_len,
                            dtype=torch.long, device=labels.device)

        # x_lens is after subsampling.  Actually we don't need it.
        (x, x_lens) = self.encoder(x, x_lens)

        logprobs = self.decoder(labels, x)
        return logprobs
