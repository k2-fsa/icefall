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


from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear
from icefall.utils import make_pad_mask, time_warp
from lhotse.dataset import SpecAugment


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
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            ScaledLinear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

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

    def forward_cr_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTC loss with consistency regularization loss.
        Args:
          encoder_out:
            Encoder output, of shape (2 * N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (2 * N,).
          targets:
            Target Tensor of shape (2 * sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC loss
        ctc_output = self.ctc_output(encoder_out)  # (2 * N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )

        # Compute consistency regularization loss
        exchanged_targets = ctc_output.detach().chunk(2, dim=0)
        exchanged_targets = torch.cat(
            [exchanged_targets[1], exchanged_targets[0]], dim=0
        )  # exchange: [x1, x2] -> [x2, x1]
        cr_loss = nn.functional.kl_div(
            input=ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )  # (2 * N, T, C)
        length_mask = make_pad_mask(encoder_out_lens).unsqueeze(-1)
        cr_loss = cr_loss.masked_fill(length_mask, 0.0).sum()

        return ctc_loss, cr_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        warmup: float = 1.0,
        use_cr_ctc: bool = False,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
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
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        if use_cr_ctc:
            if use_spec_aug:
                assert spec_augment is not None and spec_augment.time_warp_factor < 1
                # Apply time warping before input duplicating
                assert supervision_segments is not None
                x = time_warp(
                    x,
                    time_warp_factor=time_warp_factor,
                    supervision_segments=supervision_segments,
                )
                # Independently apply frequency masking and time masking to the two copies
                x = spec_augment(x.repeat(2, 1, 1))
            else:
                x = x.repeat(2, 1, 1)
            x_lens = x_lens.repeat(2)
            y = k2.ragged.cat([y, y], axis=0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(encoder_out_lens > 0)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        # Compute CTC loss
        targets = y.values
        if not use_cr_ctc:
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
            cr_loss = torch.empty(0)
        else:
            ctc_loss, cr_loss = self.forward_cr_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
            ctc_loss = ctc_loss * 0.5
            cr_loss = cr_loss * 0.5

        return ctc_loss, cr_loss
