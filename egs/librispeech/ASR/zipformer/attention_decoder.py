#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
from typing import List, Optional

import k2
import torch
import torch.nn as nn
from label_smoothing import LabelSmoothingLoss
from scaling import penalize_abs_values_gt

from icefall.utils import add_eos, add_sos, make_pad_mask


class AttentionDecoderModel(nn.Module):
    """
    Args:
        vocab_size (int): Number of classes.
        decoder_dim: (int,int): embedding dimension of 2 encoder stacks
        attention_dim: (int,int): attention dimension of 2 encoder stacks
        num_heads (int, int): number of heads
        dim_feedforward (int, int): feedforward dimension in 2 encoder stacks
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int = 512,
        num_decoder_layers: int = 6,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        sos_id: int = 1,
        eos_id: int = 1,
        dropout: float = 0.1,
        ignore_id: int = -1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.ignore_id = ignore_id

        # For the segment of the warmup period, we let the Embedding
        # layer learn something.  Then we start to warm up the other encoders.
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=decoder_dim,
            num_decoder_layers=num_decoder_layers,
            attention_dim=attention_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            memory_dim=memory_dim,
            dropout=dropout,
        )

        # Used to calculate attention-decoder loss
        self.loss_fun = LabelSmoothingLoss(
            ignore_index=ignore_id, label_smoothing=label_smoothing, reduction="sum"
        )

    def _pre_ys_in_out(self, ys: k2.RaggedTensor, ys_lens: torch.Tensor):
        """Prepare ys_in_pad and ys_out_pad."""
        ys_in = add_sos(ys, sos_id=self.sos_id)
        # [B, S+1], start with SOS
        ys_in_pad = ys_in.pad(mode="constant", padding_value=self.eos_id)
        ys_in_lens = ys_lens + 1

        ys_out = add_eos(ys, eos_id=self.eos_id)
        # [B, S+1], end with EOS
        ys_out_pad = ys_out.pad(mode="constant", padding_value=self.ignore_id)

        return ys_in_pad.to(torch.int64), ys_in_lens, ys_out_pad.to(torch.int64)

    def calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys: k2.RaggedTensor,
        ys_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate attention-decoder loss.
        Args:
          encoder_out: (batch, num_frames, encoder_dim)
          encoder_out_lens: (batch,)
          token_ids: A list of token id list.

        Return: The attention-decoder loss.
        """
        ys_in_pad, ys_in_lens, ys_out_pad = self._pre_ys_in_out(ys, ys_lens)

        # decoder forward
        decoder_out = self.decoder(
            x=ys_in_pad,
            x_lens=ys_in_lens,
            memory=encoder_out,
            memory_lens=encoder_out_lens,
        )

        loss = self.loss_fun(x=decoder_out, target=ys_out_pad)
        return loss

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        token_ids: List[List[int]],
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from attention-decoder.
        Args:
          encoder_out: (batch, num_frames, encoder_dim)
          encoder_out_lens: (batch,)
          token_ids: A list of token id list.

        Return: A tensor of shape (batch, num_tokens).
        """
        ys = k2.RaggedTensor(token_ids).to(device=encoder_out.device)
        row_splits = ys.shape.row_splits(1)
        ys_lens = row_splits[1:] - row_splits[:-1]

        ys_in_pad, ys_in_lens, ys_out_pad = self._pre_ys_in_out(ys, ys_lens)

        # decoder forward
        decoder_out = self.decoder(
            x=ys_in_pad,
            x_lens=ys_in_lens,
            memory=encoder_out,
            memory_lens=encoder_out_lens,
        )

        batch_size, _, num_classes = decoder_out.size()
        nll = nn.functional.cross_entropy(
            decoder_out.view(-1, num_classes),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        return nll


class TransformerDecoder(nn.Module):
    """Transfomer decoder module.

    Args:
        vocab_size: output dim
        d_model: decoder dimension
        num_decoder_layers: number of decoder layers
        attention_dim: total dimension of multi head attention
        num_heads: number of attention heads
        feedforward_dim: hidden dimension of feed_forward module
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_decoder_layers: int = 6,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Absolute positional encoding
        self.pos = PositionalEncoding(d_model, dropout_rate=0.1)

        self.num_layers = num_decoder_layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    memory_dim=memory_dim,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: Input tensor of shape (batch, tgt_len).
          x_lens: A tensor of shape (batch,) containing the number of tokens in `x`
            before padding.
          memory:
            Memory sequence of shape (batch, src_len, memory_dim).
          memory_lens:
            A tensor of shape (batch,) containing the number of frames in
            `memory` before padding.

        Returns:
            Decoded token logits before softmax (batch, tgt_len, vocab_size)
        """
        x = self.embed(x)  # (batch, tgt_len, embed_dim)
        x = self.pos(x)  # (batch, tgt_len, embed_dim)

        x = x.permute(1, 0, 2)  # (tgt_len, batch, embed_dim)

        # construct attn_mask for self-attn modules
        padding_mask = make_pad_mask(x_lens)  # (batch, tgt_len)
        causal_mask = subsequent_mask(x.shape[0], device=x.device)  # (seq_len, seq_len)
        attn_mask = torch.logical_or(
            padding_mask.unsqueeze(1),  # (batch, 1, seq_len)
            torch.logical_not(causal_mask).unsqueeze(0)  # (1, seq_len, seq_len)
        )  # (batch, seq_len, seq_len)

        if memory is not None:
            memory = memory.permute(1, 0, 2)  # (src_len, batch, memory_dim)
            # construct memory_attn_mask for cross-attn modules
            memory_padding_mask = make_pad_mask(memory_lens)  # (batch, src_len)
            memory_attn_mask = memory_padding_mask.unsqueeze(1)  # (batch, 1, src_len)
        else:
            memory_attn_mask = None

        for i, mod in enumerate(self.layers):
            x = mod(
                x,
                attn_mask=attn_mask,
                memory=memory,
                memory_attn_mask=memory_attn_mask,
            )

        x = x.permute(1, 0, 2)  # (batch, tgt_len, vocab_size)
        x = self.output_layer(x)

        return x


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        d_model: equal to decoder_dim, total dimension of the decoder
        attention_dim: total dimension of multi head attention
        num_heads: number of attention heads
        feedforward_dim: hidden dimension of feed_forward module
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()

        self.norm_self_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model, attention_dim, num_heads, dropout=0.0
        )

        self.norm_src_attn = nn.LayerNorm(d_model)
        self.src_attn = MultiHeadAttention(
            d_model, attention_dim, num_heads, memory_dim=memory_dim, dropout=0.0
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (seq_len, batch, embed_dim).
            attn_mask: A binary mask for self-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
            memory: Memory sequence of shape (seq_len, batch, memory_dim).
            memory_attn_mask: A binary mask for cross-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
        """
        # self-attn module
        qkv = self.norm_self_attn(x)
        self_attn_out = self.self_attn(
            query=qkv, key=qkv, value=qkv, attn_mask=attn_mask
        )
        x = x + self.dropout(self_attn_out)

        # cross-attn module
        q = self.norm_src_attn(x)
        src_attn_out = self.src_attn(
            query=q, key=memory, value=memory, attn_mask=memory_attn_mask
        )
        x = x + self.dropout(src_attn_out)

        # feed-forward module
        x = x + self.dropout(self.feed_forward(self.norm_ff(x)))

        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module, but must be a multiple of num_heads.
        num_heads: number of parallel attention heads.
        memory_dim: dimension of memory embedding, optional.
        dropout: a Dropout layer on attn_output_weights.
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        memory_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim, num_heads, attention_dim
        )
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.

        self.linear_q = nn.Linear(embed_dim, attention_dim, bias=True)
        self.linear_k = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )
        self.linear_v = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )

        self.out_proj = nn.Linear(attention_dim, embed_dim, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dot product attention.

        Args:
            query: Query tensor of shape (tgt_len, batch, embed_dim).
            key: Key tensor of shape (src_len, batch, embed_dim or memory_dim).
            value: Value tensor of shape (src_len, batch, embed_dim or memory_dim).
            key_padding_mask: A binary mask indicating which elements are padding.
                Its shape is (batch, src_len).
            attn_mask: A binary mask indicating which elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).

        Returns:
            Output tensor of shape (tgt_len, batch, embed_dim).
        """
        num_heads = self.num_heads
        head_dim = self.head_dim

        tgt_len, batch, _ = query.shape
        src_len = key.shape[0]

        q = self.linear_q(query)  # (tgt_len, batch, num_heads * head_dim)
        k = self.linear_k(key)  # (src_len, batch, num_heads * head_dim)
        v = self.linear_v(value)  # (src_len, batch, num_heads * head_dim)

        q = q.reshape(tgt_len, batch, num_heads, head_dim)
        q = q.permute(1, 2, 0, 3)  # (batch, head, tgt_len, head_dim)
        k = k.reshape(src_len, batch, num_heads, head_dim)
        k = k.permute(1, 2, 3, 0)  # (batch, head, head_dim, src_len)
        v = v.reshape(src_len, batch, num_heads, head_dim)
        v = v.reshape(src_len, batch * num_heads, head_dim).transpose(0, 1)

        # Note: could remove the scaling operation when using ScaledAdam
        # (batch, head, tgt_len, src_len)
        attn_weights = torch.matmul(q, k) / math.sqrt(head_dim)

        # From zipformer.py:
        # This is a harder way of limiting the attention scores to not be too large.
        # It incurs a penalty if any of them has an absolute value greater than 50.0.
        # this should be outside the normal range of the attention scores.  We use
        # this mechanism instead of, say, a limit on entropy, because once the entropy
        # gets very small gradients through the softmax can become very small, and
        # some mechanisms like that become ineffective.
        attn_weights = penalize_abs_values_gt(attn_weights, limit=50.0, penalty=1.0e-04)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch, src_len), key_padding_mask.shape
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"),
            )

        if attn_mask is not None:
            assert (
                attn_mask.shape == (batch, 1, src_len)
                or attn_mask.shape == (batch, tgt_len, src_len)
            ), attn_mask.shape
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        attn_weights = attn_weights.view(batch * num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # (batch * head, tgt_len, head_dim)
        attn_output = torch.bmm(attn_weights, v)
        assert attn_output.shape == (batch * num_heads, tgt_len, head_dim), attn_output.shape

        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(tgt_len, batch, num_heads * head_dim)

        # (batch, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class PositionalEncoding(nn.Module):
    """Positional encoding.
    Copied from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py#L35.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


def _test_attention_decoder_model():
    m = AttentionDecoderModel(
        vocab_size=500,
        decoder_dim=512,
        num_decoder_layers=6,
        attention_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        memory_dim=384,
        dropout=0.1,
        sos_id=1,
        eos_id=1,
        ignore_id=-1,
    )

    num_param = sum([p.numel() for p in m.parameters()])
    print(f"Number of model parameters: {num_param}")

    m.eval()
    encoder_out = torch.randn(2, 50, 384)
    encoder_out_lens = torch.full((2,), 50)
    token_ids = [[1, 2, 3, 4], [2, 3, 10]]

    nll = m.nll(encoder_out, encoder_out_lens, token_ids)
    print(nll)


if __name__ == "__main__":
    _test_attention_decoder_model()
