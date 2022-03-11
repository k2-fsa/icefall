# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


from typing import Optional

import k2
import torch
import torch.nn as nn
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
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,). It should have an attribute:
            output_dim.
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, U, C) and (N, T, U, C). Its
            output shape is also (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
          decoder_giga:
            The decoder for the GigaSpeech dataset.
          joiner_giga:
            The joiner for the GigaSpeech dataset.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        if decoder_giga is not None:
            assert hasattr(decoder_giga, "blank_id")

        self.encoder = encoder

        self.decoder = decoder
        self.joiner = joiner

        vocab_size = self.joiner.output_dim
        joiner_dim = self.joiner.input_dim

        # Note: self.joiner.output_dim is equal to vocab_size.
        # This layer is to transform the decoder output for computing
        # simple loss
        self.simple_decoder_linear = nn.Linear(
            self.decoder.embedding_dim, vocab_size
        )

        # This layer is to transform the encoder output for computing
        # simple loss
        self.simple_encoder_linear = nn.Linear(
            self.encoder.output_dim, vocab_size
        )

        # Transform the output of decoder so that it can be added
        # with the output of encoder in the joiner.
        self.decoder_linear = nn.Linear(vocab_size, joiner_dim)

        # Transform the output of encoder so that it can be added
        # with the output of decoder in the joiner
        self.encoder_linear = nn.Linear(vocab_size, joiner_dim)

        self.decoder_giga = decoder_giga
        self.joiner_giga = joiner_giga

        if decoder_giga is not None:
            self.simple_decoder_giga_linear = nn.Linear(
                self.decoder.embedding_dim, vocab_size
            )
            self.simple_encoder_giga_linear = nn.Linear(
                self.encoder.output_dim, vocab_size
            )
            self.decoder_giga_linear = nn.Linear(vocab_size, joiner_dim)
            self.encoder_giga_linear = nn.Linear(vocab_size, joiner_dim)
        else:
            self.simple_decoder_giga_linear = None
            self.simple_encoder_giga_linear = None
            self.decoder_giga_linear = None
            self.encoder_giga_linear = None

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        libri: bool = True,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
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
          libri:
            True to use the decoder and joiner for the LibriSpeech dataset.
            False to use the decoder and joiner for the GigaSpeech dataset.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        if libri:
            decoder = self.decoder
            joiner = self.joiner
            simple_decoder_linear = self.simple_decoder_linear
            simple_encoder_linear = self.simple_encoder_linear
            decoder_linear = self.decoder_linear
            encoder_linear = self.encoder_linear
        else:
            decoder = self.decoder_giga
            joiner = self.joiner_giga
            simple_decoder_linear = self.simple_decoder_giga_linear
            simple_encoder_linear = self.simple_encoder_giga_linear
            decoder_linear = self.decoder_giga_linear
            encoder_linear = self.encoder_giga_linear

        # decoder_out: [B, S + 1, C]
        decoder_out = decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (x.size(0), 4), dtype=torch.int64, device=x.device
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        simple_decoder_out = simple_decoder_linear(decoder_out)
        simple_encoder_out = simple_encoder_linear(encoder_out)

        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=simple_decoder_out,
            am=simple_encoder_out,
            symbols=y_padded,
            termination_symbol=blank_id,
            lm_only_scale=lm_scale,
            am_only_scale=am_scale,
            boundary=boundary,
            reduction="sum",
            return_grad=True,
        )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, C]
        # lm_pruned : [B, T, prune_range, C]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=simple_encoder_out, lm=simple_decoder_out, ranges=ranges
        )

        am_pruned = encoder_linear(am_pruned)
        lm_pruned = decoder_linear(lm_pruned)

        # logits : [B, T, prune_range, C]
        logits = joiner(am_pruned, lm_pruned)

        pruned_loss = k2.rnnt_loss_pruned(
            logits=logits,
            symbols=y_padded,
            ranges=ranges,
            termination_symbol=blank_id,
            boundary=boundary,
            reduction="sum",
        )

        return (simple_loss, pruned_loss)
