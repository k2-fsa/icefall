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
from chunk_decoder import ChunkDecoder
from zipformer import Zipformer2


class Zipformer2LM(nn.Module):

    def __init__(self,
                 encoder_embed: nn.Module,
                 encoder: Zipformer2,
                 decoder: ChunkDecoder):
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

        chunk_size = self.decoder.chunk_size
        labels_shifted = labels.t()  # (time, batch)
        labels_shifted = torch.cat((torch.zeros_like(labels_shifted[:chunk_size]),
                                    labels_shifted[:-chunk_size]),
                                   dim=0)

        x = self.encoder_embed(labels_shifted)
        x_lens = torch.full((batch_size,), seq_len,
                            dtype=torch.long, device=labels.device)
        # x_lens is after subsampling.  Actually we don't need it.


        (x, x_lens) = self.encoder(x, x_lens)

        logprobs = self.decoder(labels, x)
        return logprobs
