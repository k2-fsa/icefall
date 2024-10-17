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
import torch.nn.functional as F
from scaling import Balancer


class Decoder(torch.nn.Module):
    """
    This class modifies the stateless decoder from the following paper:
    RNN-transducer with stateless prediction network
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction network.
    Different from the above paper, it adds an extra Conv1d right after the embedding layer.
    """

    def __init__(
        self, vocab_size: int, decoder_dim: int, context_size: int, device: torch.device,
    ) -> None:
        """
        Decoder initialization.

        Parameters
        ----------
        vocab_size : int
            A number of tokens or modeling units, includes blank.
        decoder_dim : int
            A dimension of the decoder embeddings, and the decoder output.
        context_size : int
            A number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, decoder_dim)

        if context_size < 1:
            raise ValueError(
                'RNN-T decoder context size should be an integer greater '
                f'or equal than 1, but got {context_size}.',
            )
        self.context_size = context_size

        self.conv = torch.nn.Conv1d(
            decoder_dim,
            decoder_dim,
            context_size,
            groups=decoder_dim // 4,
            bias=False,
            device=device,
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the stateless Decoder module. Returns an output decoder tensor.

        Parameters
        ----------
        y : torch.Tensor[torch.int32]
            The input integer tensor of shape (N, context_size).
            The module input that corresponds to the last context_size decoded token indexes.

        Returns
        -------
        torch.Tensor[torch.float32]
            An output float tensor of shape (N, 1, decoder_dim).
        """

        # this stuff about clamp() is a fix for a mismatch at utterance start,
        # we use negative ids in RNN-T decoding.
        embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(2)

        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
            embedding_out = torch.nn.functional.relu(embedding_out)

        return embedding_out


class DecoderModule(torch.nn.Module):
    """
    A helper module to combine decoder, decoder projection, and joiner inference together.
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        joiner_dim: int,
        context_size: int,
        beam: int,
        device: torch.device,
    ) -> None:
        """
        DecoderModule initialization.

        Parameters
        ----------
        vocab_size:
            A number of tokens or modeling units, includes blank.
        decoder_dim : int
            A dimension of the decoder embeddings, and the decoder output.
        joiner_dim : int
            Input joiner dimension.
        context_size : int
            A number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        beam : int
            A decoder beam.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.decoder = Decoder(vocab_size, decoder_dim, context_size, device)
        self.decoder_proj = torch.nn.Linear(decoder_dim, joiner_dim, device=device)
        self.joiner = Joiner(joiner_dim, vocab_size, device)

        self.vocab_size = vocab_size
        self.beam = beam

    def forward(
        self, decoder_input: torch.Tensor, encoder_out: torch.Tensor, hyps_log_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the stateless Decoder module. Returns an output decoder tensor.

        Parameters
        ----------
        decoder_input : torch.Tensor[torch.int32]
            The input integer tensor of shape (num_hyps, context_size).
            The module input that corresponds to the last context_size decoded token indexes.
        encoder_out : torch.Tensor[torch.float32]
            An output tensor from the encoder after projection of shape (num_hyps, joiner_dim).
        hyps_log_prob : torch.Tensor[torch.float32]
            Hypothesis probabilities in a logarithmic scale of shape (num_hyps, 1).

        Returns
        -------
        torch.Tensor[torch.float32]
            A float output tensor of logit token probabilities of shape (num_hyps, vocab_size).
        """

        decoder_out = self.decoder(decoder_input)
        decoder_out = self.decoder_proj(decoder_out)

        logits = self.joiner(encoder_out, decoder_out[:, 0, :])

        tokens_log_prob = torch.log_softmax(logits, dim=1)
        log_probs = (tokens_log_prob + hyps_log_prob).reshape(-1)

        hyps_topk_log_prob, topk_indexes = log_probs.topk(self.beam)
        topk_hyp_indexes = torch.floor_divide(topk_indexes, self.vocab_size).to(torch.int32)
        topk_token_indexes = torch.remainder(topk_indexes, self.vocab_size).to(torch.int32)
        tokens_topk_prob = torch.exp(tokens_log_prob.reshape(-1)[topk_indexes])

        return hyps_topk_log_prob, tokens_topk_prob, topk_hyp_indexes, topk_token_indexes