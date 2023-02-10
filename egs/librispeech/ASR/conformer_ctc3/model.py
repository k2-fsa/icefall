# Copyright  2021-2022  Xiaomi Corp.     (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Zengwei Yao)
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


import math
from typing import Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear


class CTCModel(nn.Module):
    """It implements https://www.cs.toronto.edu/~graves/icml_2006.pdf
    "Connectionist Temporal Classification: Labelling Unsegmented
    Sequence Data with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        encoder_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          encoder_dim:
            The feature embedding dimension.
          vocab_size:
            The vocabulary size.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder = encoder
        self.ctc_output_module = nn.Sequential(
            nn.Dropout(p=0.1),
            ScaledLinear(encoder_dim, vocab_size),
        )

    def get_ctc_output(
        self,
        encoder_out: torch.Tensor,
        delay_penalty: float = 0.0,
        blank_threshold: float = 0.99,
    ):
        """Compute ctc log-prob and optionally (delay_penalty > 0) apply delay penalty.
        We first split utterance into sub-utterances according to the
        blank probs, and then add sawtooth-like "blank-bonus" values to
        the blank probs.
        See https://github.com/k2-fsa/icefall/pull/669 for details.

        Args:
          encoder_out:
            A tensor with shape of (N, T, C).
          delay_penalty:
            A constant used to scale the delay penalty score.
          blank_threshold:
            The threshold used to split utterance into sub-utterances.
        """
        output = self.ctc_output_module(encoder_out)
        log_prob = nn.functional.log_softmax(output, dim=-1)

        if self.training and delay_penalty > 0:
            T_arange = torch.arange(encoder_out.shape[1]).to(device=encoder_out.device)
            # split into sub-utterances using the blank-id
            mask = log_prob[:, :, 0] >= math.log(blank_threshold)  # (B, T)
            mask[:, 0] = True
            cummax_out = (T_arange * mask).cummax(dim=-1)[0]  # (B, T)
            # the sawtooth "blank-bonus" value
            penalty = T_arange - cummax_out  # (B, T)
            penalty_all = torch.zeros_like(log_prob)
            penalty_all[:, :, 0] = delay_penalty * penalty
            # apply latency penalty on probs
            log_prob = log_prob + penalty_all

        return log_prob

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        warmup: float = 1.0,
        delay_penalty: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          warmup: a floating point value which increases throughout training;
            values >= 1.0 are fully warmed up and have all modules present.
          delay_penalty:
            A constant used to scale the delay penalty score.
        """
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(encoder_out_lens > 0)
        nnet_output = self.get_ctc_output(encoder_out, delay_penalty=delay_penalty)
        return nnet_output, encoder_out_lens
