#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Zengwei Yao)
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
import itertools
import logging
import math
import random
from typing import List, Tuple

import k2
import torch
import torch.nn as nn
from label_smoothing import LabelSmoothingLoss
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    Identity,
    MaxEig,
    ScaledConv1d,
    ScaledLinear,
    Whiten,
    _diag,
    penalize_abs_values_gt,
    random_clamp,
    softmax,
)
from zipformer import FeedforwardModule

from icefall.utils import add_eos, add_sos, make_pad_mask


class AttentionDecoderModel(nn.Module):
    """
    Args:
        vocab_size (int): Number of classes.
        encoder_dim (int):
        d_model: (int,int): embedding dimension of 2 encoder stacks
        attention_dim: (int,int): attention dimension of 2 encoder stacks
        nhead (int, int): number of heads
        dim_feedforward (int, int): feedforward dimension in 2 encoder stacks
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        vgg_frontend (bool): whether to use vgg frontend.
        warmup_batches (float): number of batches to warm up over
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        unmasked_dim: int,
        num_decoder_layers: int,
        attention_dim: int,
        nhead: int,
        feedforward_dim: int,
        dropout: float,
        sos_id: int,
        eos_id: int,
        ignore_id: int = -1,
        warmup_batches: float = 4000.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.ignore_id = ignore_id

        # For the segment of the warmup period, we let the Embedding
        # layer learn something.  Then we start to warm up the other encoders.
        self.decoder = TransformerDecoder(
            vocab_size,
            d_model,
            unmasked_dim,
            num_decoder_layers,
            attention_dim,
            nhead,
            feedforward_dim,
            dropout,
            warmup_begin=warmup_batches * 0.5,
            warmup_end=warmup_batches * 1.0,
        )

        # Used to calculate attention-decoder loss
        self.loss_fun = LabelSmoothingLoss(
            ignore_index=ignore_id, label_smoothing=label_smoothing, reduction="sum"
        )

    def _pre_ys_in_out(self, token_ids: List[List[int]], device: torch.device):
        """Prepare ys_in_pad and ys_out_pad."""
        ys = k2.RaggedTensor(token_ids).to(device=device)

        row_splits = ys.shape.row_splits(1)
        ys_lens = row_splits[1:] - row_splits[:-1]

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
        token_ids: List[List[int]],
    ) -> torch.Tensor:
        """Calculate attention-decoder loss.
        Args:
          encoder_out: (batch, num_frames, encoder_dim)
          encoder_out_lens: (batch,)
          token_ids: A list of token id list.

        Return: The attention-decoder loss.
        """
        ys_in_pad, ys_in_lens, ys_out_pad = self._pre_ys_in_out(
            token_ids, encoder_out.device
        )

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

        Return: A tensor of shape (batch,).
        """
        ys_in_pad, ys_in_lens, ys_out_pad = self._pre_ys_in_out(
            token_ids, encoder_out.device
        )

        # decoder forward
        decoder_out = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

        batch_size, _, num_classes = decoder_out.size()
        nll = nn.functional.cross_entropy(
            decoder_out.view(-1, num_classes),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="None",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(1)
        return nll


class TransformerDecoder(nn.Module):
    """Transfomer decoder module.
    It is modified from https://github.com/espnet/espnet/blob/master/espnet2/asr/decoder/transformer_decoder.py.

    Args:
        vocab_size: output dim
        d_model: equal to encoder_dim
        num_decoder_layers: number of decoder layers
        attention_dim: total dimension of multi head attention
        n_head: number of attention heads
        feedforward_dim: hidden dimension of feed_forward module
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        unmasked_dim: int,
        num_decoder_layers: int,
        attention_dim: int,
        nhead: int,
        feedforward_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
    ):
        super().__init__()
        self.unmasked_dim = unmasked_dim

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Using absolute positional encoding
        self.pos = PositionalEncoding(d_model, dropout_rate=0.1)

        self.num_layers = num_decoder_layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, attention_dim, nhead, feedforward_dim, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

        # will be written to, see set_batch_count() Note: in inference time this
        # may be zero but should be treated as large, we can check if
        # self.training is true.
        self.batch_count = 0
        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)
        self.warmup_begin = warmup_begin
        self.warmup_end = warmup_end
        # module_seed is for when we need a random number that is unique to the module but
        # shared across jobs.   It's used to randomly select how many layers to drop,
        # so that we can keep this consistent across worker tasks (for efficiency).
        self.module_seed = torch.randint(0, 1000, ()).item()

        delta = (1.0 / num_decoder_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_decoder_layers):
            self.layers[i].warmup_begin = cur_begin
            cur_begin += delta
            self.layers[i].warmup_end = cur_begin

    def get_layers_to_drop(self, rnd_seed: int):
        ans = set()
        if not self.training:
            return ans

        batch_count = self.batch_count
        num_layers = len(self.layers)

        def get_layerdrop_prob(layer: int) -> float:
            layer_warmup_begin = self.layers[layer].warmup_begin
            layer_warmup_end = self.layers[layer].warmup_end

            initial_layerdrop_prob = 0.5
            final_layerdrop_prob = 0.05

            if batch_count == 0:
                # As a special case, if batch_count == 0, return 0 (drop no
                # layers).  This is rather ugly, I'm afraid; it is intended to
                # enable our scan_pessimistic_batches_for_oom() code to work correctly
                # so if we are going to get OOM it will happen early.
                # also search for 'batch_count' with quotes in this file to see
                # how we initialize the warmup count to a random number between
                # 0 and 10.
                return 0.0
            elif batch_count < layer_warmup_begin:
                return initial_layerdrop_prob
            elif batch_count > layer_warmup_end:
                return final_layerdrop_prob
            else:
                # linearly interpolate
                t = (batch_count - layer_warmup_begin) / layer_warmup_end
                assert 0.0 <= t < 1.001, t
                return initial_layerdrop_prob + t * (
                    final_layerdrop_prob - initial_layerdrop_prob
                )

        shared_rng = random.Random(batch_count + self.module_seed)
        independent_rng = random.Random(rnd_seed)

        layerdrop_probs = [get_layerdrop_prob(i) for i in range(num_layers)]
        tot = sum(layerdrop_probs)
        # Instead of drawing the samples independently, we first randomly decide
        # how many layers to drop out, using the same random number generator between
        # jobs so that all jobs drop out the same number (this is for speed).
        # Then we use an approximate approach to drop out the individual layers
        # with their specified probs while reaching this exact target.
        num_to_drop = int(tot) + int(shared_rng.random() < (tot - int(tot)))

        layers = list(range(num_layers))
        independent_rng.shuffle(layers)

        # go through the shuffled layers until we get the required number of samples.
        if num_to_drop > 0:
            for layer in itertools.cycle(layers):
                if independent_rng.random() < layerdrop_probs[layer]:
                    ans.add(layer)
                if len(ans) == num_to_drop:
                    break
        if shared_rng.random() < 0.005 or __name__ == "__main__":
            logging.info(
                f"warmup_begin={self.warmup_begin:.1f}, warmup_end={self.warmup_end:.1f}, "
                f"batch_count={batch_count:.1f}, num_to_drop={num_to_drop}, layers_to_drop={ans}"
            )
        return ans

    def get_feature_mask(self, x: torch.Tensor) -> float:
        # Note: The actual return type is Union[List[float], List[Tensor]],
        # but to make torch.jit.script() work, we use List[float]
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (num_frames, batch_size, encoder_dims0)
        """
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return 1.0

        batch_size, num_frames, d_model = x.size()

        feature_mask_dropout_prob = 0.15
        frame_mask = (
            torch.rand(batch_size, num_frames, 1, device=x.device)
            > feature_mask_dropout_prob
        ).to(x.dtype)

        feature_mask = torch.ones(
            batch_size, num_frames, d_model, dtype=x.dtype, device=x.device
        )
        feature_mask[:, :, self.unmasked_dim :] *= frame_mask

        return feature_mask

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = make_pad_mask(ys_in_lens)[:, None, :].to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask | (~m)

        memory = hs_pad
        memory_mask = make_pad_mask(hlens)[:, None, :].to(memory.device)

        tgt = self.embed(tgt)
        tgt = self.pos(tgt)

        rnd_seed = tgt.numel() + random.randint(0, 1000)
        layers_to_drop = self.get_layers_to_drop(rnd_seed)

        feature_mask = self.get_feature_mask(tgt)

        for i, mod in enumerate(self.layers):
            if i in layers_to_drop:
                continue
            tgt = mod(tgt, tgt_mask, memory, memory_mask)
            tgt = tgt * feature_mask

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
        d_model: int,
        attention_dim: int,
        nhead: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()

        # will be written to, see set_batch_count()
        self.batch_count = 0

        self.self_attn = MultiHeadedAttention(
            d_model, attention_dim, nhead, dropout=0.0
        )
        self.src_attn = MultiHeadedAttention(d_model, attention_dim, nhead, dropout=0.0)
        self.feed_forward = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.norm_final = BasicNorm(d_model)

        self.bypass_scale = nn.Parameter(torch.tensor(0.5))

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.whiten = Whiten(
            num_groups=1, whitening_limit=5.0, prob=(0.025, 0.25), grad_scale=0.01
        )

    def get_bypass_scale(self):
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return self.bypass_scale
        if random.random() < 0.1:
            # ensure we get grads if self.bypass_scale becomes out of range
            return self.bypass_scale
        # hardcode warmup period for bypass scale
        warmup_period = 20000.0
        initial_clamp_min = 0.75
        final_clamp_min = 0.25
        if self.batch_count > warmup_period:
            clamp_min = final_clamp_min
        else:
            clamp_min = initial_clamp_min - (self.batch_count / warmup_period) * (
                initial_clamp_min - final_clamp_min
            )
        return self.bypass_scale.clamp(min=clamp_min, max=1.0)

    def get_dynamic_dropout_rate(self):
        # return dropout rate for the dynamic modules (self_attn, src_attn, feed_forward); this
        # starts at 0.2 and rapidly decreases to 0.  Its purpose is to keep the training stable
        # at the beginning, by making the network focus on the feedforward modules.
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return 0.0
        warmup_period = 2000.0
        initial_dropout_rate = 0.2
        final_dropout_rate = 0.0
        if self.batch_count > warmup_period:
            return final_dropout_rate
        else:
            return initial_dropout_rate - (
                initial_dropout_rate * final_dropout_rate
            ) * (self.batch_count / warmup_period)

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
        tgt_orig = tgt

        # dropout rate for submodules that interact with time.
        dynamic_dropout = self.get_dynamic_dropout_rate()

        # self-attn module
        if random.random() >= dynamic_dropout:
            tgt = tgt + self.self_attn(tgt, tgt, tgt, tgt_mask)

        # cross-attn module
        if random.random() >= dynamic_dropout:
            tgt = tgt + self.src_attn(tgt, memory, memory, memory_mask)

        # feed-forward module
        tgt = tgt + self.feed_forward(tgt)

        tgt = self.norm_final(self.balancer(tgt))

        delta = tgt - tgt_orig
        tgt = tgt_orig + delta * self.get_bypass_scale()

        return self.whiten(tgt)


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
        self, embed_dim: int, attention_dim: int, num_heads: int, dropout: float = 0.0
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = attention_dim // num_heads
        assert self.head_dim % 2 == 0, self.head_dim
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim,
            num_heads,
            attention_dim,
        )

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5, dividing it between the query and key.
        self.linear_q = ScaledLinear(
            embed_dim, attention_dim, bias=True, initial_scale=self.head_dim**-0.25
        )
        self.linear_k = ScaledLinear(
            embed_dim, attention_dim, bias=True, initial_scale=self.head_dim**-0.25
        )
        self.linear_v = ScaledLinear(
            embed_dim,
            attention_dim // 2,
            bias=True,
            initial_scale=self.head_dim**-0.25,
        )

        # self.whiten_v is applied on the values in forward();
        # it just copies the keys but prevents low-rank distribution by modifying grads.
        self.whiten_v = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )
        self.whiten_k = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # the following are for diagnosics only, see --print-diagnostics option.
        # they only copy their inputs.
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

        self.out_proj = ScaledLinear(
            attention_dim // 2, embed_dim, bias=True, initial_scale=0.05
        )

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

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_k(k)  # does nothing in the forward pass.
        v = self.whiten_v(v)  # does nothing in the forward pass.

        q = q.reshape(bsz, tgt_len, num_heads, head_dim)
        q = q.transpose(1, 2)  # (batch, head, time1, head_dim)
        k = k.reshape(bsz, src_len, num_heads, head_dim)
        k = k.permute(0, 2, 3, 1)  # (batch, head, head_dim, time2)
        v = v.reshape(bsz, src_len, num_heads, head_dim // 2)
        v = v.transpose(1, 2).reshape(bsz * num_heads, src_len, head_dim // 2)

        # (batch, head, time1, time2)
        attn_output_weights = torch.matmul(q, k)

        if mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                mask.unsqueeze(1), float("-inf")
            )

        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )

        # Using this version of softmax, defined in scaling.py,
        # should save a little of the memory used in backprop by, if
        # we are in automatic mixed precision mode (amp) == autocast,
        # only storing the half-precision output for backprop purposes.
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        # (bsz * head, time1, head_dim_v)
        attn_output = torch.bmm(attn_output_weights, v)
        assert attn_output.shape == (bsz * num_heads, tgt_len, head_dim // 2)
        attn_output = (
            attn_output.reshape(bsz, num_heads, tgt_len, head_dim // 2)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, self.attention_dim // 2)
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
        d_model=384,
        unmasked_dim=256,
        num_decoder_layers=6,
        attention_dim=192,
        nhead=8,
        feedforward_dim=2048,
        dropout=0.1,
        sos_id=1,
        eos_id=1,
        ignore_id=-1,
    )
    m.eval()
    encoder_out = torch.randn(2, 50, 384)
    encoder_out_lens = torch.full((2,), 50)
    token_ids = [[1, 2, 3, 4], [2, 3, 10]]
    loss = m.calc_att_loss(encoder_out, encoder_out_lens, token_ids)
    print(loss)


if __name__ == "__main__":
    _test_attention_decoder_model()
