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

from typing import List

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from transformer import encoder_padding_mask

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.utils import encode_supervisions


class CTCModel(nn.Module):
    """It implements a CTC model with an auxiliary attention head."""

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        encoder_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            An instance of `EncoderInterface`. The shared encoder for the CTC and attention
            branches
          decoder:
            An instance of `nn.Module`. This is the decoder for the attention branch.
          encoder_dim:
            Dimension of the encoder output.
          decoder_dim:
            Dimension of the decoder output.
          vocab_size:
            Number of tokens of the modeling unit including blank.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder = encoder
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )
        self.decoder = decoder

    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        supervisions: torch.Tensor,
        graph_compiler: BpeCtcTrainingGraphCompiler,
        subsampling_factor: int = 1,
        beam_size: int = 10,
        reduction: str = "sum",
        use_double_scores: bool = False,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Tensor of dimension (N, T, C) where N is the batch size,
            T is the number of frames, and C is the feature dimension.
          x_lens:
            Tensor of dimension (N,) where N is the batch size.
          supervisions:
            Supervisions are used in training.
          graph_compiler:
            It is used to compile a decoding graph from texts.
          subsampling_factor:
            It is used to compute the `supervisions` for the encoder.
          beam_size:
            Beam size used in `k2.ctc_loss`.
          reduction:
            Reduction method used in `k2.ctc_loss`.
          use_double_scores:
            If True, use double precision in `k2.ctc_loss`.
        Returns:
          Return the CTC loss, attention loss, and the total number of frames.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        nnet_output, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)
        # compute ctc log-probs
        ctc_output = self.ctc_output(nnet_output)

        # NOTE: We need `encode_supervisions` to sort sequences with
        # different duration in decreasing order, required by
        # `k2.intersect_dense` called in `k2.ctc_loss`
        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=subsampling_factor
        )
        num_frames = supervision_segments[:, 2].sum().item()

        # Works with a BPE model
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)

        dense_fsa_vec = k2.DenseFsaVec(
            ctc_output,
            supervision_segments.cpu(),
            allow_truncate=subsampling_factor - 1,
        )

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=beam_size,
            reduction=reduction,
            use_double_scores=use_double_scores,
        )

        if self.decoder is not None:
            nnet_output = nnet_output.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            mmodel = (
                self.decoder.module if hasattr(self.decoder, "module") else self.decoder
            )
            # Note: We need to generate an unsorted version of token_ids
            # `encode_supervisions()` called above sorts text, but
            # encoder_memory and memory_mask are not sorted, so we
            # use an unsorted version `supervisions["text"]` to regenerate
            # the token_ids
            #
            # See https://github.com/k2-fsa/icefall/issues/97
            # for more details
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            mask = encoder_padding_mask(nnet_output.size(0), supervisions)
            mask = mask.to(nnet_output.device) if mask is not None else None
            att_loss = mmodel.forward(
                nnet_output,
                mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        else:
            att_loss = torch.tensor([0])

        return ctc_loss, att_loss, num_frames
