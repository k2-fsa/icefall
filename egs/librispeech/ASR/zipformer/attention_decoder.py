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

# The model structure is modified from Daniel Povey's Zipformer
# https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7/zipformer.py

import math
from typing import List, Optional, Tuple

import k2
import torch
import torch.nn as nn

from label_smoothing import LabelSmoothingLoss
from icefall.utils import add_eos, add_sos, make_pad_mask
from scaling import penalize_abs_values_gt


class AttentionDecoderModel(nn.Module):
    """
    Args:
        vocab_size (int): Number of classes.
        decoder_dim: (int,int): embedding dimension of 2 encoder stacks
        attention_dim: (int,int): attention dimension of 2 encoder stacks
        nhead (int, int): number of heads
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
        nhead: int = 8,
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
            nhead=nhead,
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
        decoder_out = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

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
        decoder_out = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

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
    It is modified from https://github.com/espnet/espnet/blob/master/espnet2/asr/decoder/transformer_decoder.py.

    Args:
        vocab_size: output dim
        d_model: decoder dimension
        num_decoder_layers: number of decoder layers
        attention_dim: total dimension of multi head attention
        n_head: number of attention heads
        feedforward_dim: hidden dimension of feed_forward module
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_decoder_layers: int = 6,
        attention_dim: int = 512,
        nhead: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Using absolute positional encoding
        self.pos = PositionalEncoding(d_model, dropout_rate=0.1)

        self.num_layers = num_decoder_layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model, attention_dim, nhead, feedforward_dim, memory_dim, dropout
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        memory_lens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, (batch, maxlen_in, feat)
            memory_lens: (batch,)
            ys_in_pad: input token ids, (batch, maxlen_out)
            ys_in_lens: (batch, )

        Returns:
            tgt: decoded token score before softmax (batch, maxlen_out, vocab_size)
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = make_pad_mask(ys_in_lens)[:, None, :].to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask | (~m)

        memory_mask = make_pad_mask(memory_lens)[:, None, :].to(memory.device)

        tgt = self.embed(tgt)
        tgt = self.pos(tgt)

        for i, mod in enumerate(self.layers):
            tgt = mod(tgt, tgt_mask, memory, memory_mask)

        tgt = self.output_layer(tgt)
        return tgt


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        d_model: equal to encoder_dim
        attention_dim: total dimension of multi head attention
        n_head: number of attention heads
        feedforward_dim: hidden dimension of feed_forward module
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        attention_dim: int = 512,
        nhead: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()

        self.norm_self_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadedAttention(d_model, attention_dim, nhead, dropout=0.0)

        self.norm_src_attn = nn.LayerNorm(d_model)
        self.src_attn = MultiHeadedAttention(d_model, attention_dim, nhead, memory_dim=memory_dim, dropout=0.0)

        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
        """
        # self-attn module
        tgt_norm = self.norm_self_attn(tgt)
        tgt = tgt + self.dropout(self.self_attn(tgt_norm, tgt_norm, tgt_norm, tgt_mask))

        # cross-attn module
        tgt = tgt + self.dropout(self.src_attn(self.norm_src_attn(tgt), memory, memory, memory_mask))

        # feed-forward module
        tgt = tgt + self.dropout(self.feed_forward(self.norm_ff(tgt)))

        return tgt


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module, may be less or more than embed_dim
            but must be a multiple of num_heads.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        memory_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim,
            num_heads,
            attention_dim,
        )

        self.linear_q = nn.Linear(embed_dim, attention_dim, bias=True)
        self.linear_k = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )
        self.linear_v = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )
        self.scale = math.sqrt(self.head_dim)

        self.out_proj = nn.Linear(attention_dim, embed_dim, bias=True)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        bsz, tgt_len, _ = query.size()
        src_len = key.size(1)
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        q = q.reshape(bsz, tgt_len, num_heads, head_dim)
        q = q.transpose(1, 2)  # (batch, head, time1, head_dim)
        k = k.reshape(bsz, src_len, num_heads, head_dim)
        k = k.permute(0, 2, 3, 1)  # (batch, head, head_dim, time2)
        v = v.reshape(bsz, src_len, num_heads, head_dim)
        v = v.transpose(1, 2).reshape(bsz * num_heads, src_len, head_dim)

        # (batch, head, time1, time2)
        attn_output_weights = torch.matmul(q, k) / self.scale

        # attn_output_weights = torch.matmul(q, k)
        # # This is a harder way of limiting the attention scores to not be too large.
        # # It incurs a penalty if any of them has an absolute value greater than 50.0.
        # # this should be outside the normal range of the attention scores.  We use
        # # this mechanism instead of, say, a limit on entropy, because once the entropy
        # # gets very small gradients through the softmax can become very small, and
        # # some mechanisms like that become ineffective.
        attn_output_weights = penalize_abs_values_gt(
            attn_output_weights, limit=50.0, penalty=1.0e-04
        )

        if mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                mask.unsqueeze(1), float("-inf")
            )

        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        # (bsz * head, time1, head_dim_v)
        attn_output = torch.bmm(attn_output_weights, v)
        assert attn_output.shape == (bsz * num_heads, tgt_len, head_dim)
        attn_output = (
            attn_output.reshape(bsz, num_heads, tgt_len, head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, self.attention_dim)
        )
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
        nhead=8,
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
