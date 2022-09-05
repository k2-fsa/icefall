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


from typing import Tuple

import torch
import torch.nn as nn


class GradientFilterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        batch_dim: int,  # e.g., 1
        threshold: float,  # e.g., 10.0
    ) -> torch.Tensor:
        if x.requires_grad:
            if batch_dim < 0:
                batch_dim += x.ndim
            ctx.batch_dim = batch_dim
            ctx.threshold = threshold
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        dim = ctx.batch_dim
        if x_grad.shape[dim] == 1:
            return x_grad, None, None
        norm_dims = [d for d in range(x_grad.ndim) if d != dim]
        norm_of_batch = x_grad.norm(dim=norm_dims, keepdim=True)
        norm_of_batch_sorted = norm_of_batch.sort(dim=dim)[0]
        median_idx = (x_grad.shape[dim] - 1) // 2
        median_norm = norm_of_batch_sorted.narrow(
            dim=dim, start=median_idx, length=1
        )
        mask = norm_of_batch <= ctx.threshold * median_norm
        return x_grad * mask, None, None


class GradientFilter(torch.nn.Module):
    """This is used to filter out elements that have extremely large gradients
    in batch.

    Args:
      batch_dim (int):
        The batch dimension.
      threshold (float):
        For each element in batch, its gradient will be
        filtered out if the gradient norm is larger than
        `grad_norm_threshold * median`, where `median` is the median
        value of gradient norms of all elememts in batch.
    """

    def __init__(self, batch_dim: int = 1, threshold: float = 10.0):
        super(GradientFilter, self).__init__()
        self.batch_dim = batch_dim
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientFilterFunction.apply(
            x,
            self.batch_dim,
            self.threshold,
        )


class TdnnLstm(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 3,
        grad_norm_threshold: float = 10.0,
    ) -> None:
        """
        Args:
          num_features:
            The input dimension of the model.
          num_classes:
            The output dimension of the model.
          subsampling_factor:
            It reduces the number of output frames by this factor.
          grad_norm_threshold:
            For each sequence element in batch, its gradient will be
            filtered out if the gradient norm is larger than
            `grad_norm_threshold * median`, where `median` is the median
            value of gradient norms of all elememts in batch.
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=500,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(
                in_channels=500,
                out_channels=500,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(
                in_channels=500,
                out_channels=500,
                kernel_size=3,
                stride=self.subsampling_factor,  # stride: subsampling_factor!
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
        )
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(input_size=500, hidden_size=500, num_layers=1)
                for _ in range(5)
            ]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=500, affine=False) for _ in range(5)]
        )
        self.grad_filters = nn.ModuleList(
            [GradientFilter(batch_dim=1, threshold=grad_norm_threshold)]
        )

        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=500, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            Its shape is [N, C, T]

        Returns:
          The output tensor has shape [N, T, C]
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (N, C, T) -> (T, N, C) -> how LSTM expects it
        for lstm, bnorm, grad_filter in zip(
            self.lstms, self.lstm_bnorms, self.grad_filters
        ):
            x_new, _ = lstm(grad_filter(x))
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
