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
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.utils import AttributeDict
from scaling import ScaledLinear
from typing import List, Union


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
    ):
        output = self.ctc_output_module(encoder_out)
        log_prob = nn.functional.log_softmax(output, dim=-1)
        return log_prob

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        supervision_segments: torch.Tensor,
        texts: List[str],
        graph_compiler: Union[
            BpeCtcTrainingGraphCompiler, CtcTrainingGraphCompiler
        ],
        params: AttributeDict,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          supervision_segments:
            The supervision tensor has shape ``(batch_size, 3)``.
            Its second dimension contains information about sequence index [0],
            start frames [1] and num frames [2].
          texts:
            A list of transcription strings.
          graph_compiler:
            It is used to build a decoding graph from a ctc topo and training
            transcript. The training transcript is contained in the given `batch`,
            while the ctc topo is built when this compiler is instantiated.
          params:
            Parameters for training. See :func:`get_params`.
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
        Returns:
          Return the ctc loss.
        """
        assert params.reduction in ("sum", "none"), params.reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.size(0) == x_lens.size(0)

        encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        # calculate ctc loss
        nnet_output = self.get_ctc_output(encoder_out)

        if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
            # Works with a BPE model
            token_ids = graph_compiler.texts_to_ids(texts)
            decoding_graph = graph_compiler.compile(token_ids)
        elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
            # Works with a phone lexicon
            decoding_graph = graph_compiler.compile(texts)
        else:
            raise ValueError(
                f"Unsupported type of graph compiler: {type(graph_compiler)}"
            )

        dense_fsa_vec = k2.DenseFsaVec(
            nnet_output,
            supervision_segments,
            allow_truncate=params.subsampling_factor - 1,
        )

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=params.beam_size,
            reduction=params.reduction,
            use_double_scores=params.use_double_scores,
        )

        return ctc_loss
