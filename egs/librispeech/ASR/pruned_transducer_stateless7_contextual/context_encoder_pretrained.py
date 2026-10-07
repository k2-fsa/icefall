# Copyright    2024  (authors: Ruizhe Huang)
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

import torch
from context_encoder import ContextEncoder


class ContextEncoderPretrained(ContextEncoder):
    """Maps a pretrained word embedding (e.g., BERT or fastText) of each
    biasing word to the context embedding space with a small MLP."""

    def __init__(self, context_encoder_dim: int, output_dim: int):
        """
        Args:
          context_encoder_dim:
            Dimension of the pretrained word embeddings.
          output_dim:
            Dimension of the output context embeddings.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(context_encoder_dim, 256)
        self.linear3 = torch.nn.Linear(256, 256)
        self.linear4 = torch.nn.Linear(256, 256)
        self.linear2 = torch.nn.Linear(256, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        word_list: torch.Tensor,
        word_lengths: Optional[list] = None,
        is_encoder_side: Optional[bool] = None,
    ) -> torch.Tensor:
        out = self.sigmoid(self.linear1(word_list))
        out = self.sigmoid(self.linear3(out))
        out = self.sigmoid(self.linear4(out))
        return self.linear2(out)
