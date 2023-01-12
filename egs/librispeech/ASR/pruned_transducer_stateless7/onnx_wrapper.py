# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from torch import nn


class TritonOnnxDecoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:
        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419
    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.
    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(self, model):
        """
        Args:
           model: zipformer decoder model
        """
        super().__init__()

        self.model = model

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        need_pad = False
        return self.model(y, need_pad)


class TritonOnnxJoiner(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.model = model
        self.encoder_proj = model.encoder_proj
        self.decoder_proj = model.decoder_proj

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        project_input = False
        return self.model(encoder_out, decoder_out, project_input)
