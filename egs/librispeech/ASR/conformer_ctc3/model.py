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


import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.utils import AttributeDict
from scaling import ScaledLinear
from typing import List, Tuple, Union


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
        warmup: float = 1.0,
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
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(encoder_out_lens > 0)
        nnet_output = self.get_ctc_output(encoder_out)
        return nnet_output, encoder_out_lens
