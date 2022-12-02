import torch
import torch.nn as nn
import torch.nn.functional as F


class LConv(nn.Module):
    """A convolution module to prevent information loss.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        bias: bool = True,
    ):
        """
        Args:
          channels:
            Dimension of the input embedding, and of the lconv output.
        """
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.depthwise_conv = nn.Conv1d(
            2 * channels,
            2 * channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=2 * channels,
            bias=bias,
        )

        self.pointwise_conv2 = nn.Conv1d(
            2 * channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A 3-D tensor of shape (N, T, C).
        Returns:
            Return a tensor of shape (N, T, C).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(0, 2, 1)  # (#batch, channels, time).

        x = self.pointwise_conv1(x) # (batch, 2*channels, time)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv2(x) # (batch, channels, time)

        return x.permute(0, 2, 1)