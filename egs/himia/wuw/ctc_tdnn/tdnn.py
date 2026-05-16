#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (Author: Liyong Guo)
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

from torch import nn, Tensor


class Tdnn(nn.Module):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=240,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240, out_channels=240, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=240, affine=False),
            nn.Conv1d(
                in_channels=240,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.LogSoftmax(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (N, T, C).
        Returns:
            Tensor: Predictor tensor of dimension (N, T, C).
        """

        x = x.transpose(1, 2)
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        return x
