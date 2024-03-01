from typing import Optional

import torch
from flow import Transpose


class DurationDiscriminator(torch.nn.Module):  # vits2
    def __init__(
        self,
        channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        eps: float = 1e-5,
        global_channels: int = -1,
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.global_channels = global_channels

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.conv_1 = torch.nn.Conv1d(
            channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.LayerNorm(
                hidden_channels,
                eps=eps,
                elementwise_affine=True,
            ),
            Transpose(1, 2),
        )

        self.conv_2 = torch.nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )

        self.norm_2 = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.LayerNorm(
                hidden_channels,
                eps=eps,
                elementwise_affine=True,
            ),
            Transpose(1, 2),
        )

        self.dur_proj = torch.nn.Conv1d(1, hidden_channels, 1)

        self.pre_out_conv_1 = torch.nn.Conv1d(
            2 * hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )

        self.pre_out_norm_1 = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.LayerNorm(
                hidden_channels,
                eps=eps,
                elementwise_affine=True,
            ),
            Transpose(1, 2),
        )

        self.pre_out_conv_2 = torch.nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )

        self.pre_out_norm_2 = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.LayerNorm(
                hidden_channels,
                eps=eps,
                elementwise_affine=True,
            ),
            Transpose(1, 2),
        )

        if global_channels > 0:
            self.cond_layer = torch.nn.Conv1d(global_channels, channels, 1)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 1), torch.nn.Sigmoid()
        )

    def forward_probability(
        self, x: torch.Tensor, x_mask: torch.Tensor, dur: torch.Tensor
    ):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.dropout(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = self.dropout(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)

        return output_prob

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        dur_r: torch.Tensor,
        dur_hat: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ):

        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond_layer(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.dropout(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.dropout(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur)
            output_probs.append(output_prob)

        return output_probs
