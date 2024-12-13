import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def window_sumsquare(
    window: torch.Tensor,
    n_samples: int,
    hop_length: int = 256,
    win_length: int = 1024,
):
    """
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_samples : int > 0
        The number of expected samples.
    hop_length : int > 0
        The number of samples to advance between frames
    win_length :
        The length of the window function.
    Returns
    -------
    wss : torch.Tensor, The sum-squared envelope of the window function.
    """

    n_frames = (n_samples - win_length) // hop_length + 1
    output_size = (n_frames - 1) * hop_length + win_length
    device = window.device

    # Window envelope
    window_sq = window.square().expand(1, n_frames, -1).transpose(1, 2)
    window_envelope = torch.nn.functional.fold(
        window_sq,
        output_size=(1, output_size),
        kernel_size=(1, win_length),
        stride=(1, hop_length),
    ).squeeze()
    window_envelope = torch.nn.functional.pad(
        window_envelope, (0, n_samples - output_size)
    )
    return window_envelope


class ISTFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        padding: str = "none",
        window_type: str = "povey",
        max_samples: int = 1440000,  # 1440000 / 24000 = 60s
    ):
        super(ISTFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        assert filter_length >= win_length
        # Consistence with lhotse, search "create_frame_window" in https://github.com/lhotse-speech/lhotse
        assert window_type in [
            "hanning",
            "povey",
        ], f"Only 'hanning' and 'povey' windows are supported, given {window_type}."
        fft_window = torch.hann_window(win_length, periodic=False)
        if window_type == "povey":
            fft_window = fft_window.pow(0.85)

        if filter_length > win_length:
            pad_size = (filter_length - win_length) // 2
            fft_window = torch.nn.functional.pad(fft_window, (pad_size, pad_size))

        window_sum = window_sumsquare(
            window=fft_window,
            n_samples=max_samples,
            hop_length=hop_length,
            win_length=filter_length,
        )

        inverse_basis *= fft_window

        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window)
        self.register_buffer("window_sum", window_sum)
        self.tiny = torch.finfo(torch.float16).tiny

    def forward(self, magnitude, phase):
        magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        inverse_transform = F.conv_transpose1d(
            magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )
        inverse_transform = inverse_transform.squeeze(1)

        window_sum = self.window_sum
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if self.window_sum.size(-1) < inverse_transform.size(-1):
                logging.warning(
                    f"The precomputed `window_sumsquare` is too small, recomputing, "
                    f"from {self.window_sum.size(-1)} to {inverse_transform.size(-1)}"
                )
                window_sum = window_sumsquare(
                    window=self.fft_window,
                    n_samples=inverse_transform.size(-1),
                    win_length=self.filter_length,
                    hop_length=self.hop_length,
                )
        window_sum = window_sum[: inverse_transform.size(-1)]
        approx_nonzero_indices = (window_sum > self.tiny).nonzero().squeeze()

        inverse_transform[:, approx_nonzero_indices] /= window_sum[
            approx_nonzero_indices
        ]

        # scale by hop ratio
        inverse_transform *= float(self.filter_length) / self.hop_length
        assert self.padding in ["none", "same", "center"]
        if self.padding == "center":
            pad_len = self.filter_length // 2
        elif self.padding == "same":
            pad_len = (self.filter_length - self.hop_length) // 2
        else:
            return inverse_transform
        return inverse_transform[:, pad_len:-pad_len]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class Generator(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = 80,
        dim: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        padding: str = "none",
        max_samples: int = 1440000,  # 1440000 / 24000 = 60s
    ):
        super(Generator, self).__init__()
        self.feature_dim = feature_dim
        self.embed = nn.Conv1d(feature_dim, dim, kernel_size=7, padding=3)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        layer_scale_init_value = 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

        self.out_proj = torch.nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding=padding,
            max_samples=max_samples,
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.out_proj(x).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        # safeguard to prevent excessively large magnitudes
        mag = torch.clip(mag, max=1e2)
        audio = self.istft(mag, phase)
        return audio
