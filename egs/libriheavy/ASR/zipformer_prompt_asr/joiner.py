# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        context_dim: int = 512,
        context_injection: bool = False,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim, initial_scale=0.25)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)
        if context_injection:
            self.context_proj = ScaledLinear(
                context_dim, joiner_dim, initial_scale=0.25
            )
        else:
            self.context_proj = None

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        context: torch.Tensor = None,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
          context:
            An embedding vector representing the previous context information
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim == 4
        assert encoder_out.shape[:-1] == decoder_out.shape[:-1]

        if project_input:
            if context:
                logit = (
                    self.encoder_proj(encoder_out)
                    + self.decoder_proj(decoder_out)
                    + self.context_proj(context)
                )
            else:
                logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            if context is not None:
                logit = encoder_out + decoder_out + context.unsqueeze(1).unsqueeze(1)
            else:
                logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit
