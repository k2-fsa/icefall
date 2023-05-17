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

class Decoder(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 vocab_size: int):
        """
        A 'decoder' that computes the probability of symbols in a language modeling task.
        """
        super().__init__()


        self.to_hidden = nn.Linear(
            embed_dim, hidden_dim, bias=False,
        )
        # no padding, will manually pad on the left so it is causal.
        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            groups=hidden_dim,
            kernel_size=3
        )
        self.activation = nn.Tanh()
        self.hidden_to_vocab = nn.Linear(
            hidden_dim, vocab_size,
        )

        self.bypass = nn.Linear(
            embed_dim, vocab_size, bias=False,
        )


    def forward(self,
                labels: Tensor,
                encoder_embed: Tensor) -> Tensor:
        """
        Compute log-probs.
        Args:
           labels: the labels, a Tensor of integer type of shape (batch_size, seq_len);
        encoder_embed: the embeddings from the encoder, of shape (seq_len, batch_size, embed_dim)

        Returns:
            returns the log-probs for each symbol, in a Tensor of shape (batch_size, seq_len).
        """
        (batch_size, seq_len) = labels.shape
        (num_chunks, _batch_size, embed_dim) = encoder_embed.shape

        assert batch_size == _batch_size

        bypass = self.bypass(encoder_embed)

        x = self.to_hidden(encoder_embed)  # (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 2, 0)  # (N,C,H) = (batch_size, hidden_dim, seq_len)
        x = torch.nn.functional.pad(x, (2, 0)) # pad left with 2 frames.
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        x = self.activation(x)
        x = self.hidden_to_vocab(x) # (batch_size, seq_len, vocab_size)

        x = x + bypass.transpose(0, 1)
        x = x.log_softmax(dim=-1)

        logprobs = torch.gather(x, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

        return logprobs
