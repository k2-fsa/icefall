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

import math

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import add_sos


def reverse_label_smoothing(
    logprobs: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    This function is written by Dan.

    Modifies `logprobs` in such a way that if you compute a data probability
    using `logprobs`, it will be equivalent to a label-smoothed data probability
    with the supplied label-smoothing constant alpha (e.g. alpha=0.1).
    This allows us to use `logprobs` in things like RNN-T and CTC and
    get a kind of label-smoothed version of those sequence objectives.

    Label smoothing means that if the reference label is i, we convert it
    into a distribution with weight (1-alpha) on i, and alpha distributed
    equally to all labels (including i itself).

    Note: the output logprobs can be interpreted as cross-entropies, meaning
    we correct for the entropy of the smoothed distribution.

    Args:
      logprobs:
        A Tensor of shape (*, num_classes), containing logprobs that sum
        to one: e.g. the output of log_softmax.
      alpha:
        A constant that defines the extent of label smoothing, e.g. 0.1.

    Returns:
      modified_logprobs, a Tensor of shape (*, num_classes), containing
      "fake" logprobs that will give you label-smoothed probabilities.
    """
    assert alpha >= 0.0 and alpha < 1
    if alpha == 0.0:
        return logprobs
    num_classes = logprobs.shape[-1]

    # We correct for the entropy of the label-smoothed target distribution, so
    # the resulting logprobs can be thought of as cross-entropies, which are
    # more interpretable.
    #
    # The expression for entropy below is not quite correct -- it treats
    # the target label and the smoothed version of the target label as being
    # separate classes -- but this can be thought of as an adjustment
    # for the way we compute the likelihood below, which also treats the
    # target label and its smoothed version as being separate.
    target_entropy = -(
        (1 - alpha) * math.log(1 - alpha)
        + alpha * math.log(alpha / num_classes)
    )
    sum_logprob = logprobs.sum(dim=-1, keepdim=True)

    return (
        logprobs * (1 - alpha) + sum_logprob * (alpha / num_classes)
    ) + target_entropy


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
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
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        label_smoothing_factor: float,
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
          label_smoothing_factor:
            The factor for label smoothing. Should be in the range [0, 1).
        Returns:
          Return the transducer loss.
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

        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        decoder_out = self.decoder(sos_y_padded)

        # +1 here since a blank is prepended to each utterance.
        logits = self.joiner(
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            encoder_out_len=x_lens,
            decoder_out_len=y_lens + 1,
        )
        # logits is of shape (sum_all_TU, vocab_size)

        log_probs = logits.log_softmax(dim=-1)
        log_probs = reverse_label_smoothing(log_probs, label_smoothing_factor)

        # rnnt_loss requires 0 padded targets
        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)

        # We don't put this `import` at the beginning of the file
        # as it is required only in the training, not during the
        # reference stage
        import optimized_transducer

        loss = optimized_transducer.transducer_loss(
            logits=log_probs,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
            from_log_softmax=True,
        )

        return loss
