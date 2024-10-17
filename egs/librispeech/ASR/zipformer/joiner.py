# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
import torch.nn as nn
from scaling import ScaledLinear


class Joiner(torch.nn.Module):

    def __init__(self, joiner_dim: int, vocab_size: int, device: torch.device) -> None:
        """
        Joiner initialization.

        Parameters
        ----------
        joiner_dim : int
            Input joiner dimension.
        vocab_size : int
            Output joiner dimension, the vocabulary size, the number of BPEs of the model.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.output_linear = torch.nn.Linear(joiner_dim, vocab_size, device=device)

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the Joiner module. Returns an output tensor after a simple joining.

        Parameters
        ----------
        encoder_out : torch.Tensor[torch.float32]
            An output tensor from the encoder after projection of shape (N, joiner_dim).
        decoder_out : torch.Tensor[torch.float32]
            An output tensor from the decoder after projection of shape (N, joiner_dim).

        Returns
        -------
        torch.Tensor[torch.float32]
            A float output tensor of log token probabilities of shape (N, vocab_size).
        """

        return self.output_linear(torch.tanh(encoder_out + decoder_out))
