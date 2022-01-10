#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
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


import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from conv1d_abs_attention import Conv1dAbs
from transformer import Supervisions, Transformer, encoder_padding_mask


class Conformer(Transformer):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): attention dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimention
        num_encoder_layers (int): number of encoder layers
        num_decoder_layers (int): number of decoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        normalize_before (bool): whether to use layer_norm before the first block.
        vgg_frontend (bool): whether to use vgg frontend.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,  # 12
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        vgg_frontend: bool = False,
        use_feat_batchnorm: bool = False,
    ) -> None:
        super(Conformer, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            normalize_before=normalize_before,
            vgg_frontend=vgg_frontend,
            use_feat_batchnorm=use_feat_batchnorm,
        )

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = ConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            cnn_module_kernel,
            normalize_before,
        )
        self.encoder = ConformerEncoder(encoder_layer, num_encoder_layers)
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(d_model)
        else:
            # Note: TorchScript detects that self.after_norm could be used inside forward()
            #       and throws an error without this change.
            self.after_norm = identity

    def run_encoder(
        self, x: Tensor, supervisions: Optional[Supervisions] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
          x:
            The model input. Its shape is (N, T, C).
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.

        Returns:
            Tensor: Predictor tensor of dimension (input_length, batch_size, d_model).
            Tensor: Mask tensor of dimension (batch_size, input_length)
        """
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        mask = encoder_padding_mask(x.size(0), supervisions)
        if mask is not None:
            mask = mask.to(x.device)
        # print("encoder input: ", x[0][0][:20])
        x = self.encoder(x, pos_emb, src_key_padding_mask=mask)  # (T, B, F)

        if self.normalize_before:
            x = self.after_norm(x)

        return x, mask


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
        normalize_before: whether to use layer_norm before the first block.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)

        self.norm_ff_macaron = nn.LayerNorm(
            d_model
        )  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module

        # define layernorm for conv1d_abs
        self.norm_conv_abs = nn.LayerNorm(d_model)
        self.layernorm = nn.LayerNorm(d_model)

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(
            d_model
        )  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

        self.kernel_size = 21
        self.padding = int((self.kernel_size - 1) / 2)
        self.conv1d_channels = 768
        self.linear1 = nn.Linear(512, self.conv1d_channels)
        self.conv1d_abs = Conv1dAbs(
            self.conv1d_channels,
            self.conv1d_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.linear2 = nn.Linear(self.conv1d_channels, 512)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """

        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(
            self.feed_forward_macaron(src)
        )
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)

        inf = torch.tensor(float("inf"), device=src.device)

        def check_inf(x):
            if x.max() == inf:
                print("Error: inf found: ", x)
                assert 0

                # modified-attention module

        residual = src
        if self.normalize_before:
            src = self.norm_conv_abs(src)

        src = self.linear1(src * 0.25)
        src = torch.exp(src.clamp(min=-75, max=75))
        check_inf(src)
        src = src.permute(1, 2, 0)
        src = self.conv1d_abs(src) / self.kernel_size
        check_inf(src)
        src = src.permute(2, 0, 1)
        src = torch.log(src.clamp(min=1e-20))
        check_inf(src)
        src = self.linear2(src)
        src = self.layernorm(src)
        # multipy the output by 0.5 later.
        # do a comparison.

        src = residual + self.dropout(src)
        if not self.normalize_before:
            src = self.norm_conv_abs(src)

        # convolution module
        residual = src
        if self.normalize_before:
            src = self.norm_conv(src)
        src = residual + self.dropout(self.conv_module(src))
        if not self.normalize_before:
            src = self.norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)

        return src


class ConformerEncoder(nn.TransformerEncoder):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None
    ) -> None:
        super(ConformerEncoder, self).__init__(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for mod in self.layers:
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick as in "T
        # ransformer-XL:Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self, channels: int, kernel_size: int, bias: bool = True
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x: Tensor) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def identity(x):
    return x
