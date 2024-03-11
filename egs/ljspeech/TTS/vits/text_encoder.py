#!/usr/bin/env python3
# Copyright           2023  Xiaomi Corp.        (authors: Zengwei Yao)
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

"""Text encoder module in VITS.

This code is based on
  - https://github.com/jaywalnut310/vits
  - https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/vits/text_encoder.py
  - https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer_stateless/conformer.py
"""

import copy
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from icefall.utils import is_jit_tracing, make_pad_mask


class TextEncoder(torch.nn.Module):
    """Text encoder module in VITS.

    This is a module of text encoder described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`.
    """

    def __init__(
        self,
        vocabs: int,
        d_model: int = 192,
        num_heads: int = 2,
        dim_feedforward: int = 768,
        cnn_module_kernel: int = 5,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize TextEncoder module.

        Args:
            vocabs (int): Vocabulary size.
            d_model (int): attention dimension
            num_heads (int): number of attention heads
            dim_feedforward (int): feedforward dimention
            cnn_module_kernel (int): convolution kernel size
            num_layers (int): number of encoder layers
            dropout (float): dropout rate
        """
        super().__init__()
        self.d_model = d_model

        # define modules
        self.emb = torch.nn.Embedding(vocabs, d_model)
        torch.nn.init.normal_(self.emb.weight, 0.0, d_model**-0.5)

        # We use conformer as text encoder
        self.encoder = Transformer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            cnn_module_kernel=cnn_module_kernel,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.proj = torch.nn.Conv1d(d_model, d_model * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input index tensor (B, T_text).
            x_lengths (Tensor): Length tensor (B,).

        Returns:
            Tensor: Encoded hidden representation (B, embed_dim, T_text).
            Tensor: Projected mean tensor (B, embed_dim, T_text).
            Tensor: Projected scale tensor (B, embed_dim, T_text).
            Tensor: Mask tensor for input tensor (B, 1, T_text).

        """
        # (B, T_text, embed_dim)
        x = self.emb(x) * math.sqrt(self.d_model)

        assert x.size(1) == x_lengths.max().item()

        # (B, T_text)
        pad_mask = make_pad_mask(x_lengths)

        # encoder assume the channel last (B, T_text, embed_dim)
        x = self.encoder(x, key_padding_mask=pad_mask)
        # Note: attention_dim == embed_dim

        # convert the channel first (B, embed_dim, T_text)
        x = x.transpose(1, 2)
        non_pad_mask = (~pad_mask).unsqueeze(1)
        stats = self.proj(x) * non_pad_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)

        return x, m, logs, non_pad_mask


class Transformer(nn.Module):
    """
    Args:
        d_model (int): attention dimension
        num_heads (int): number of attention heads
        dim_feedforward (int): feedforward dimention
        cnn_module_kernel (int): convolution kernel size
        num_layers (int): number of encoder layers
        dropout (float): dropout rate
    """

    def __init__(
        self,
        d_model: int = 192,
        num_heads: int = 2,
        dim_feedforward: int = 768,
        cnn_module_kernel: int = 5,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            cnn_module_kernel=cnn_module_kernel,
            dropout=dropout,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.after_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, key_padding_mask: Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          lengths:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        """
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        x = self.encoder(x, pos_emb, key_padding_mask=key_padding_mask)  # (T, N, C)

        x = self.after_norm(x)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward.

    Args:
        d_model: the number of expected features in the input.
        num_heads: the number of heads in the multi-head attention models.
        dim_feedforward: the dimension of the feed-forward network model.
        dropout: the dropout value (default=0.1).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        cnn_module_kernel: int,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.self_attn = RelPositionMultiheadAttention(
            d_model, num_heads, dropout=dropout
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm_ff_macaron = nn.LayerNorm(d_model)  # for the macaron style FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module
        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(d_model)  # for the final output of the block
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module

        self.ff_scale = 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the transformer encoder layer.

        Args:
            src: the sequence to the encoder layer, of shape (seq_len, batch_size, embed_dim).
            pos_emb: Positional embedding tensor, of shape (1, seq_len*2-1, pos_dim).
            key_padding_mask: the mask for the src keys per batch, of shape (batch_size, seq_len)
        """
        # macaron style feed-forward module
        src = src + self.ff_scale * self.dropout(
            self.feed_forward_macaron(self.norm_ff_macaron(src))
        )

        # multi-head self-attention module
        src_attn = self.self_attn(
            self.norm_mha(src),
            pos_emb=pos_emb,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout(src_attn)

        # convolution module
        src = src + self.dropout(self.conv_module(self.norm_conv(src)))

        # feed-forward module
        src = src + self.dropout(self.feed_forward(self.norm_ff(src)))

        src = self.norm_final(src)

        return src


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer class.
        num_layers: the number of sub-encoder-layers in the encoder.
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder layer, of shape (seq_len, batch_size, embed_dim).
            pos_emb: Positional embedding tensor, of shape (1, seq_len*2-1, pos_dim).
            key_padding_mask: the mask for the src keys per batch, of shape (batch_size, seq_len)
        """
        output = src

        for layer_index, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                key_padding_mask=key_padding_mask,
            )

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

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        x_size = x.size(1)
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size, self.d_model)
        pe_negative = torch.zeros(x_size, self.d_model)
        position = torch.arange(0, x_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
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


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: Tensor) -> Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, seq_len, 2*seq_len-1).

        Returns:
            Tensor: tensor of shape (batch, head, seq_len, seq_len)
        """
        (batch_size, num_heads, seq_len, n) = x.shape

        if not is_jit_tracing():
            assert n == 2 * seq_len - 1, f"{n} == 2 * {seq_len} - 1"

        if is_jit_tracing():
            rows = torch.arange(start=seq_len - 1, end=-1, step=-1)
            cols = torch.arange(seq_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols

            x = x.reshape(-1, n)
            x = torch.gather(x, dim=1, index=indexes)
            x = x.reshape(batch_size, num_heads, seq_len, seq_len)
            return x
        else:
            # Note: TorchScript requires explicit arg for stride()
            batch_stride = x.stride(0)
            head_stride = x.stride(1)
            time_stride = x.stride(2)
            n_stride = x.stride(3)
            return x.as_strided(
                (batch_size, num_heads, seq_len, seq_len),
                (batch_stride, head_stride, time_stride - n_stride, n_stride),
                storage_offset=n_stride * (seq_len - 1),
            )

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, (1, 2*seq_len-1, pos_dim)
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
                Its shape is (batch_size, seq_len).

        Outputs:
            A tensor of shape (seq_len, batch_size, embed_dim).
        """
        seq_len, batch_size, _ = x.shape
        scaling = float(self.head_dim) ** -0.5

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.contiguous().view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = k.contiguous().view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = (
            v.contiguous()
            .view(seq_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        q = q.transpose(0, 1)  # (batch_size, seq_len, num_head, head_dim)

        p = self.linear_pos(pos_emb).view(
            pos_emb.size(0), -1, self.num_heads, self.head_dim
        )
        # (1, 2*seq_len, num_head, head_dim) -> (1, num_head, head_dim, 2*seq_len-1)
        p = p.permute(0, 2, 3, 1)

        # (batch_size, num_head, seq_len, head_dim)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch_size, num_head, head_dim, seq_len)
        matrix_ac = torch.matmul(
            q_with_bias_u, k
        )  # (batch_size, num_head, seq_len, seq_len)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p
        )  # (batch_size, num_head, seq_len, 2*seq_len-1)
        matrix_bd = self.rel_shift(
            matrix_bd
        )  # (batch_size, num_head, seq_len, seq_len)

        # (batch_size, num_head, seq_len, seq_len)
        attn_output_weights = (matrix_ac + matrix_bd) * scaling
        attn_output_weights = attn_output_weights.view(
            batch_size * self.num_heads, seq_len, seq_len
        )

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, seq_len)
            attn_output_weights = attn_output_weights.view(
                batch_size, self.num_heads, seq_len, seq_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                batch_size * self.num_heads, seq_len, seq_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        # (batch_size * num_head, seq_len, head_dim)
        attn_output = torch.bmm(attn_output_weights, v)
        assert attn_output.shape == (
            batch_size * self.num_heads,
            seq_len,
            self.head_dim,
        )

        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, batch_size, self.embed_dim)
        )
        # (seq_len, batch_size, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
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

        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        x = self.depthwise_conv(x)
        # x is (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def _test_text_encoder():
    vocabs = 500
    d_model = 192
    batch_size = 5
    seq_len = 100

    m = TextEncoder(vocabs=vocabs, d_model=d_model)
    x, m, logs, mask = m(
        x=torch.randint(low=0, high=vocabs, size=(batch_size, seq_len)),
        x_lengths=torch.full((batch_size,), seq_len),
    )
    print(x.shape, m.shape, logs.shape, mask.shape)


if __name__ == "__main__":
    _test_text_encoder()
