# from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/wavenet/wavenet.py

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""WaveNet modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class WaveNet(torch.nn.Module):
    """WaveNet with global conditioning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        base_dilation: int = 2,
        residual_channels: int = 64,
        aux_channels: int = -1,
        gate_channels: int = 128,
        skip_channels: int = 64,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_first_conv: bool = False,
        use_last_conv: bool = False,
        scale_residual: bool = False,
        scale_skip_connect: bool = False,
    ):
        """Initialize WaveNet module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            base_dilation (int): Base dilation factor.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for local conditioning feature.
            global_channels (int): Number of channels for global conditioning feature.
            dropout_rate (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.
            use_first_conv (bool): Whether to use the first conv layers.
            use_last_conv (bool): Whether to use the last conv layers.
            scale_residual (bool): Whether to scale the residual outputs.
            scale_skip_connect (bool): Whether to scale the skip connection outputs.

        """
        super().__init__()
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.use_first_conv = use_first_conv
        self.use_last_conv = use_last_conv
        self.scale_skip_connect = scale_skip_connect

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        if self.use_first_conv:
            self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = base_dilation ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                global_channels=global_channels,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bias=bias,
                scale_residual=scale_residual,
            )
            self.conv_layers += [conv]

        # define output layers
        if self.use_last_conv:
            self.last_conv = torch.nn.Sequential(
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T) if use_first_conv else
                (B, residual_channels, T).
            x_mask (Optional[Tensor]): Mask tensor (B, 1, T).
            c (Optional[Tensor]): Local conditioning features (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning features (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T) if use_last_conv else
                (B, residual_channels, T).

        """
        # encode to hidden representation
        if self.use_first_conv:
            x = self.first_conv(x)

        # residual block
        skips = 0.0
        for f in self.conv_layers:
            x, h = f(x, x_mask=x_mask, c=c, g=g)
            skips = skips + h
        x = skips
        if self.scale_skip_connect:
            x = x * math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        if self.use_last_conv:
            x = self.last_conv(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(
        layers: int,
        stacks: int,
        kernel_size: int,
        base_dilation: int,
    ) -> int:
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [base_dilation ** (i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self) -> int:
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size, self.base_dilation
        )


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        """Initialize 1x1 Conv1d module."""
        super().__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        dilation: int = 1,
        bias: bool = True,
        scale_residual: bool = False,
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Number of local conditioning channels.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            scale_residual (bool): Whether to scale the residual outputs.

        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.scale_residual = scale_residual

        # check
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert gate_channels % 2 == 0

        # dilation conv
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # global conditioning
        if global_channels > 0:
            self.conv1x1_glo = Conv1d1x1(global_channels, gate_channels, bias=False)
        else:
            self.conv1x1_glo = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2

        # NOTE(kan-bayashi): concat two convs into a single conv for the efficiency
        #   (integrate res 1x1 + skip 1x1 convs)
        self.conv1x1_out = Conv1d1x1(
            gate_out_channels, residual_channels + skip_channels, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            x_mask Optional[torch.Tensor]: Mask tensor (B, 1, T).
            c (Optional[Tensor]): Local conditioning tensor (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv(x)

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        # global conditioning
        if g is not None:
            g = self.conv1x1_glo(g)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ga, xb + gb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # residual + skip 1x1 conv
        x = self.conv1x1_out(x)
        if x_mask is not None:
            x = x * x_mask

        # split integrated conv results
        x, s = x.split([self.residual_channels, self.skip_channels], dim=1)

        # for residual connection
        x = x + residual
        if self.scale_residual:
            x = x * math.sqrt(0.5)

        return x, s
