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
from typing import Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from subsampling import Conv2dSubsampling, VggSubsampling
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmEncoder(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        output_dim: int,
        subsampling_factor: int = 4,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        vgg_frontend: bool = False,
        proj_size: int = 0,
    ):
        super().__init__()
        real_hidden_size = proj_size if proj_size > 0 else hidden_size
        assert (
            subsampling_factor == 4
        ), "Only subsampling_factor==4 is supported at present"

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        if vgg_frontend:
            self.encoder_embed = VggSubsampling(num_features, real_hidden_size)
        else:
            self.encoder_embed = Conv2dSubsampling(num_features, real_hidden_size)

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            bias=True,
            proj_size=proj_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(real_hidden_size, output_dim),
        )

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
        """
        x = self.encoder_embed(x)

        # Caution: We assume the subsampling factor is 4!
        lengths = ((x_lens - 1) // 2 - 1) // 2
        assert x.size(1) == lengths.max().item(), (
            x.size(1),
            lengths.max(),
        )

        if False:
            # It is commented out as DDP complains that not all parameters are
            # used. Need more checks later for the reason.
            #
            # Caution: We assume the dataloader returns utterances with
            # duration being sorted in decreasing order
            packed_x = pack_padded_sequence(
                input=x,
                lengths=lengths.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )

            packed_rnn_out, _ = self.rnn(packed_x)
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
        else:
            rnn_out, _ = self.rnn(x)

        logits = self.encoder_output_layer(rnn_out)
        return logits, lengths
