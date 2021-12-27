# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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


class CombineNet(nn.Module):
    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        """
        Args:
          num_features:
            The input dimension of the audio encoder.
          num_classes:
            The output dimension of the combinenet model.
          subsampling_factor:
            It reduces the number of output frames by this factor.
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor

        # the audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=self.subsampling_factor,  # stride: subsampling_factor!
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
        )

        # the video encoder
        self.video_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 5, 5),
                stride=(1, 1, 1),
                padding=(1, 2, 2),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(
                in_channels=64,
                out_channels=96,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.linear_visual = nn.Linear(96 * 4 * 8, 512)

        # the audio-visual combining encoder based on GRU
        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=512 * 2,
                    hidden_size=512,
                    num_layers=1,
                    bidirectional=True,
                )
                for _ in range(4)
            ]
        )
        self.gru_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=1024, affine=False) for _ in range(4)]
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(
            in_features=512 * 2, out_features=self.num_classes
        )

    def forward(self, x_v, x_a):
        """
        Args:
          x_v:
            Its shape is [N, 3, H, W]
          x_a:
            Its shape is [N, C, T]
        Returns:
          The output tensor has shape [N, T, C]
        """
        x_v = self.video_encoder(x_v)
        x_v = x_v.permute(2, 0, 1, 3, 4).contiguous()
        x_v = x_v.view(x_v.size(0), x_v.size(1), -1)
        x_v = self.linear_visual(x_v)
        x_a = self.audio_encoder(x_a)

        x_v = x_v.permute(1, 0, 2)
        x_a = x_a.permute(0, 2, 1)

        # Repeat the visual features
        # to cat with the audio features in time axis.
        x_v_copy = x_v
        x_v_stack = torch.stack((x_v, x_v_copy), dim=2)
        x_v = x_v_stack.view(
            x_v_stack.size(0), 2 * x_v_stack.size(1), x_v_stack.size(3)
        )

        x = torch.cat((x_v, x_a), dim=2)
        x = x.permute(1, 0, 2)  # (N, C, T) -> (T, N, C) -> how GRU expects it
        for gru, bnorm in zip(self.grus, self.gru_bnorms):
            x_new, _ = gru(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, N, C) -> (N, C, T) -> (T, N, C)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, N, C) -> (N, T, C) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = nn.functional.log_softmax(x, dim=-1)
        return x
