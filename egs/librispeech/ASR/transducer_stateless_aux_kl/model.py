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

import random
from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        decoder_giga: Optional[nn.Module] = None,
        joiner_giga: Optional[nn.Module] = None,
        aux_module: Optional[nn.Module] = None,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, C) and (N, U, C). Its
            output shape is (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
          decoder_giga:
            The decoder for the GigaSpeech dataset.
          joiner_giga:
            The joiner for the GigaSpeech dataset.
          aux_module:
            Optional. The auxiliary branch for computing auxiliary losses.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        if decoder_giga is not None:
            assert hasattr(decoder_giga, "blank_id")

        self.encoder = encoder

        self.decoder = decoder
        self.joiner = joiner

        self.decoder_giga = decoder_giga
        self.joiner_giga = joiner_giga

        self.aux_module = aux_module

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        libri: bool = True,
        modified_transducer_prob: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
          libri:
            True to use the decoder and joiner for the LibriSpeech dataset.
            False to use the decoder and joiner for the GigaSpeech dataset.
          modified_transducer_prob:
            The probability to use modified transducer loss.
        Returns:
          Return a tuple of 3 scalar tensors containing:
            - transducer loss
            - auxiliary transducer loss
            - KL loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens, aux_input = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        if libri:
            decoder = self.decoder
            joiner = self.joiner
        else:
            decoder = self.decoder_giga
            joiner = self.joiner_giga

        decoder_out = decoder(sos_y_padded)

        # +1 here since a blank is prepended to each utterance.
        logits = joiner(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            encoder_out_len=x_lens,
            decoder_out_len=y_lens + 1,
        )

        # rnnt_loss requires 0 padded targets
        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)

        # We don't put this `import` at the beginning of the file
        # as it is required only in the training, not during the
        # reference stage
        import optimized_transducer

        assert 0 <= modified_transducer_prob <= 1

        if modified_transducer_prob == 0:
            one_sym_per_frame = False
        elif random.random() < modified_transducer_prob:
            # random.random() returns a float in the range [0, 1)
            one_sym_per_frame = True
        else:
            one_sym_per_frame = False

        log_probs = F.log_softmax(logits, dim=-1)
        transducer_loss = optimized_transducer.transducer_loss(
            logits=log_probs,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
            one_sym_per_frame=one_sym_per_frame,
            from_log_softmax=True,
        )

        aux_output = self.aux_module(aux_input)

        # Now process the auxiliary branch
        aux_logits = joiner(
            encoder_out=aux_output,
            decoder_out=decoder_out,
            encoder_out_len=x_lens,
            decoder_out_len=y_lens + 1,
        )
        aux_log_probs = F.log_softmax(aux_logits, dim=-1)

        aux_transducer_loss = optimized_transducer.transducer_loss(
            logits=aux_log_probs,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
            one_sym_per_frame=one_sym_per_frame,
            from_log_softmax=True,
        )
        kl_loss_1 = F.kl_div(
            input=log_probs,
            target=aux_log_probs,
            reduction="sum",
            log_target=True,
        )
        kl_loss_2 = F.kl_div(
            input=aux_log_probs,
            target=log_probs,
            reduction="sum",
            log_target=True,
        )
        kl_loss = (kl_loss_1 + kl_loss_2) * 0.5

        return transducer_loss, aux_transducer_loss, kl_loss
