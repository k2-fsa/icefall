#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)


import torch
import torch.nn as nn


class Tdnn(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        """
        Args:
          num_features:
            Model input dimension.
          num_classes:
            Model output dimension
        """
        super().__init__()

        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=32,
                kernel_size=3,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=32, affine=False),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=32, affine=False),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                dilation=4,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=32, affine=False),
        )
        self.output_linear = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The input tensor with shape [N, T, C]

        Returns:
          The output tensor has shape [N, T, C]
        """
        x = x.permute(0, 2, 1)  # [N, T, C] -> [N, C, T]
        x = self.tdnn(x)
        x = x.permute(0, 2, 1)  # [N, C, T] -> [N, T, C]
        x = self.output_linear(x)
        x = nn.functional.log_softmax(x, dim=-1)
        return x


def test_tdnn():
    num_features = 23
    num_classes = 4
    model = Tdnn(num_features=num_features, num_classes=num_classes)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    N = 2
    T = 100
    C = num_features
    x = torch.randn(N, T, C)
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test_tdnn()
