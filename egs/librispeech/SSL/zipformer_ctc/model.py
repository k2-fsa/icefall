# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao,
#                                                       Yifan Yang)
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

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from scaling import ScaledLinear

from icefall.utils import add_sos


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_dim: int = 768,
        vocab_size: int = 500,
    ):
        """CTC ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)

        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            inputs: `x` of (N, T, encoder_dim).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
        """
        super().__init__()

        self.encoder = encoder

        # Modules for CTC head
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward_encoder(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 2-D tensor of shape (N, T).

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        if padding_mask is None:
            padding_mask = torch.zeros_like(x, dtype=torch.bool)

        encoder_out, padding_mask = self.encoder.extract_features(
            source=x,
            padding_mask=padding_mask,
            mask=self.encoder.training,
        )
        encoder_out_lens = torch.sum(~padding_mask, dim=1)
        assert torch.all(encoder_out_lens > 0), encoder_out_lens

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward(
        self,
        x: torch.Tensor,
        y: k2.RaggedTensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, T).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
        Returns:
          Return the CTC loss,
        """
        assert x.ndim == 2, x.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == y.dim0, (x.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, padding_mask)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        # Compute CTC loss
        targets = y.values
        ctc_loss = self.forward_ctc(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            targets=targets,
            target_lengths=y_lens,
        )

        return ctc_loss, encoder_out_lens
