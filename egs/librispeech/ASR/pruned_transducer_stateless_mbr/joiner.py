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

import torch
import torch.nn as nn
from scaling import ScaledLinear

from icefall.utils import is_jit_tracing


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)
        self.output_linear_wer = ScaledLinear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        extra_output: bool = False,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
          extra_output:
            If true, return an extra tensor with shape (N, T, s_range, C) which
            will be used to calculate the `delta_wer`.
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          If extra_output is False, return a tensor of shape (N, T, s_range, C).
          If extra_output is True, return two tensors of the same shape
          (N, T, s_range, C). The two tensors produced by two different
          projection layers, one is used as the regular joiner output, the other
          is used to calculate the `delta_wer` needed by MBR training.
        """
        if not is_jit_tracing():
            assert encoder_out.ndim == decoder_out.ndim
            assert encoder_out.shape == decoder_out.shape

        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(
                decoder_out
            )
        else:
            logit = encoder_out + decoder_out

        logit = torch.tanh(logit)

        joiner_logit = self.output_linear(logit)

        if not extra_output:
            return joiner_logit
        else:
            wer_logit = self.output_linear_wer(logit)
            return joiner_logit, wer_logit
