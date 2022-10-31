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


import k2
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
        blank_threshold: float = 0.99,
        penalty_gamma: float = 0.0,
    ):
        output = self.ctc_output_module(encoder_out)
        prob = output.softmax(dim=-1)

        if penalty_gamma > 0:
            T_arange = torch.arange(encoder_out.shape[1]).to(
                device=encoder_out.device
            )
            # split into sub-utterances using the blank-id
            mask = prob[:, :, 0] >= blank_threshold  # (B, T)
            mask[:, 0] = True
            cummax_out = (T_arange * mask).cummax(dim=-1)[0]  # (B, T)
            # the sawtooth "blank-bonus" value
            penalty = T_arange - cummax_out  # (B, T)
            penalty_all = torch.zeros_like(prob)
            penalty_all[:, :, 0] = penalty_gamma * penalty
            # apply latency penalty
            prob = prob + penalty_all

        log_prob = prob.log()
        return log_prob

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        warmup: float = 1.0,
        reduction: str = "sum",
        blank_threshold=0.99,
        penalty_gamma=0.005,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
          blank_threshold:
            The threshold value used to split the utterance into sub-utterances
            for delay penalty.
          penalty_gamma:
            The factor used to times the delay penalty score.
            If set to 0, will not apply delay penalty.
        Returns:
          Return the ctc loss.
        """
        assert reduction in ("sum", "none"), reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        # calculate ctc loss
        nnet_output = self.get_ctc_output(
            encoder_out,
            blank_threshold=blank_threshold,
            penalty_gamma=penalty_gamma,
        )

        targets = []
        target_lengths = []
        for t in y.tolist():
            target_lengths.append(len(t))
            targets.extend(t)

        targets = torch.tensor(
            targets,
            device=x.device,
            dtype=torch.int64,
        )
        target_lengths = torch.tensor(
            target_lengths,
            device=x.device,
            dtype=torch.int64,
        )

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=nnet_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=x_lens,
            target_lengths=target_lengths,
            reduction=reduction,
        )

        return ctc_loss
