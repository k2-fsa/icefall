import copy
import logging
import random
from typing import Optional, Tuple

import torch
from scaling import (
    Balancer,
    FloatLike,
    Identity,
    ScaledLinear,
    ScheduledFloat,
    Whiten,
    penalize_abs_values_gt,
    softmax,
)
from torch import Tensor, nn
from zipformer import CompactRelPositionalEncoding, _whitening_schedule
from icefall.utils import make_pad_mask


class CrossAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (am_seq_len, batch_size, embed_dim) = x.shape
        (_, _, lm_seq_len, _) = attn_weights.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, lm_seq_len, am_seq_len)

        x = self.in_proj(x)  # (am_seq_len, batch_size, num_heads * value_head_dim)
        # print("projected x.shape", x.shape)

        x = x.reshape(am_seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, am_seq_len, value_head_dim)
        # print("permuted x.shape", x.shape)

        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, lm_seq_len, value_head_dim)
        # print("attended x.shape", x.shape)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(lm_seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (lm_seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)
        # print("returned x.shape", x.shape)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: input tensor, of shape (seq_len, batch_size, embed_dim)
            attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
              with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
              attn_weights.sum(dim=-1) == 1.
            cached_val: cached attention value tensor of left context,
              of shape (left_context_len, batch_size, value_dim)
            left_context_len: number of left context frames.

        Returns:
           - attention weighted output, a tensor with the same shape as x.
           - updated cached attention value tensor of left context.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        seq_len2 = seq_len + left_context_len
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len2)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)

        # Pad cached left contexts
        assert cached_val.shape[0] == left_context_len, (
            cached_val.shape[0],
            left_context_len,
        )
        x = torch.cat([cached_val, x], dim=0)
        # Update cached left contexts
        cached_val = x[-left_context_len:, ...]

        x = x.reshape(seq_len2, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)

        return x, cached_val


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
        self,
        lm_embed_dim: int = 512,
        am_embed_dim: int = 512,
        pos_dim: int = 192,
        num_heads: int = 5,
        query_head_dim: int = 32,
        pos_head_dim: int = 4,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),
    ) -> None:
        super().__init__()
        self.lm_embed_dim = lm_embed_dim
        self.am_embed_dim = am_embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_lm_dim = (query_head_dim + pos_head_dim) * num_heads
        in_am_dim = key_head_dim * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_lm_proj = ScaledLinear(
            lm_embed_dim, in_lm_dim, bias=True, initial_scale=query_head_dim**-0.25
        )
        self.in_am_proj = ScaledLinear(
            am_embed_dim, in_am_dim, bias=True, initial_scale=query_head_dim**-0.25
        )

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=0.0,
            max_abs=100.0,
            prob=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )

        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(
        self,
        lm_pruned: Tensor,
        am_pruned: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            lm_pruned: input of shape (lm_seq_len, batch_size * prune_range, decoder_embed_dim)
            am_pruned: input of shape (am_seq_len, batch_size * prune_range, encoder_embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*lm_seq_len - 1, pos_dim)
            key_padding_mask: a bool tensor of shape (batch_size * prune_range, am_seq_len).  Positions
               that are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len)
            or (seq_len, batch_size * prune_range, batch_size * prune_range),
               interpreted as ([seq_len,] batch_size * prune_range, batch_size * prune_range)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape
                                (num_heads, seq_len, batch_size * prune_range, batch_size * prune_range)
        """
        lm_pruned = self.in_lm_proj(lm_pruned)  # lm_pruned as query
        am_pruned = self.in_am_proj(am_pruned)  # am_pruned as key
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        (
            lm_seq_len,
            b_p_dim,
            _,
        ) = lm_pruned.shape  # actual dim: (lm_seq_len, batch * prune_range, _)
        (
            am_seq_len,
            _,
            _,
        ) = am_pruned.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = lm_pruned[..., 0:query_dim]  # (lm_seq_len, batch * prune_range, query_dim)
        k = am_pruned  # (am_seq_len, batch * prune_range, query_dim)
        # p is the position-encoding query
        p = lm_pruned[
            ..., query_dim:
        ]  # (lm_seq_len, batch * prune_range, pos_head_dim * num_heads)
        assert p.shape[-1] == num_heads * pos_head_dim

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(lm_seq_len, b_p_dim, num_heads, query_head_dim)
        p = p.reshape(lm_seq_len, b_p_dim, num_heads, pos_head_dim)
        k = k.reshape(am_seq_len, b_p_dim, num_heads, query_head_dim)

        # time1 refers to target (query: lm), time2 refers to source (key: am).
        q = q.permute(
            2, 1, 0, 3
        )  # (head, batch * prune_range, lm_seq_len, query_head_dim)
        p = p.permute(
            2, 1, 0, 3
        )  # (head, batch * prune_range, lm_seq_len, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch * prune_range, d_k, am_seq_len)

        attn_scores = torch.matmul(q, k)  # (head, batch, lm_seq_len, am_seq_len)

        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # We can't put random.random() in the same line
            use_pos_scores = True
        elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * lm_seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
                2, 0, 3, 1
            )
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            if torch.jit.is_tracing():
                (num_heads, b_p_dim, time1, n) = pos_scores.shape
                rows = torch.arange(start=time1 - 1, end=-1, step=-1)
                cols = torch.arange(lm_seq_len)
                rows = rows.repeat(b_p_dim * num_heads).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads, b_p_dim, time1, lm_seq_len)
            else:
                pos_scores = pos_scores.as_strided(
                    (num_heads, b_p_dim, lm_seq_len, lm_seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (lm_seq_len - 1),
                )
            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(
                attn_scores, limit=25.0, penalty=1.0e-04, name=self.name
            )
        assert attn_scores.shape == (num_heads, b_p_dim, lm_seq_len, am_seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                b_p_dim,
                am_seq_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, lm_seq_len, am_seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .mean(dim=(1, 2))
                )
                logging.info(
                    f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}"
                )


class AlignmentAttentionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        pos_dim: int = 192,
        num_heads: int = 5,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=dropout,
        )
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            value_head_dim=value_head_dim,
        )
        self.pos_encode = CompactRelPositionalEncoding(
            embed_dim=pos_dim, dropout_rate=0.15
        )

    def forward(self, am_pruned: Tensor, lm_pruned: Tensor) -> Tensor:
        if len(am_pruned.shape) == 4 and len(lm_pruned.shape) == 4:
            # src_key_padding_mask = make_pad_mask(am_pruned_lens)

            # am_pruned : [B, am_T, prune_range, encoder_dim]
            # lm_pruned : [B, lm_T, prune_range, decoder_dim]
            (batch_size, am_T, prune_range, encoder_dim) = am_pruned.shape
            (batch_size, lm_T, prune_range, decoder_dim) = lm_pruned.shape

            # merged_am_pruned : [am_T, B * prune_range, encoder_dim]
            # merged_lm_pruned : [lm_T, B * prune_range, decoder_dim]
            merged_am_pruned = am_pruned.permute(1, 0, 2, 3).reshape(
                am_T, batch_size * prune_range, encoder_dim
            )
            merged_lm_pruned = lm_pruned.permute(1, 0, 2, 3).reshape(
                lm_T, batch_size * prune_range, decoder_dim
            )
            pos_emb = self.pos_encode(merged_lm_pruned)

            attn_weights = self.cross_attn_weights(
                merged_lm_pruned, merged_am_pruned, pos_emb
            )
            # (num_heads, b_p_dim, lm_seq_len, am_seq_len)
            # print("attn_weights.shape", attn_weights.shape)
            label_level_am_representation = self.cross_attn(
                merged_am_pruned, attn_weights
            )
            # print(
            #     "label_level_am_representation.shape",
            #     label_level_am_representation.shape,
            # )
            # (lm_seq_len, batch_size * prune_range, encoder_dim)

            return label_level_am_representation.reshape(
                lm_T, batch_size, prune_range, encoder_dim
            ).permute(1, 0, 2, 3)
        elif len(am_pruned.shape) == 3 and len(lm_pruned.shape) == 3:
            # am_pruned : [am_T, B, encoder_dim]
            # lm_pruned : [lm_T, B, decoder_dim]
            (am_T, batch_size, encoder_dim) = am_pruned.shape
            (lm_T, batch_size, decoder_dim) = lm_pruned.shape

            pos_emb = self.pos_encode(lm_pruned)

            attn_weights = self.cross_attn_weights(lm_pruned, am_pruned, pos_emb)
            label_level_am_representation = self.cross_attn(am_pruned, attn_weights)
            # (T, batch_size, encoder_dim)

            return label_level_am_representation
        else:
            raise NotImplementedError("Dim Error")


if __name__ == "__main__":
    attn = AlignmentAttentionModule()

    print("__main__ === for inference ===")
    # am : [T, B, encoder_dim]
    # lm : [1, B, decoder_dim]
    am = torch.rand(100, 2, 512)
    lm = torch.rand(1, 2, 512)
    # q / K separate seq_len

    # weights = RelPositionMultiheadAttentionWeights()
    # attn = CrossAttention(512, 5, 12)
    # attn_weights = weights(lm, am, pos_emb)
    # print("weights(am_pruned, lm_pruned, pos_emb).shape", attn_weights.shape)
    # res = attn(am, attn_weights)
    res = attn(am, lm)
    print("__main__ res", res.shape)

    print("__main__ === for training ===")
    # am_pruned : [B, T, prune_range, encoder_dim]
    # lm_pruned : [B, T, prune_range, decoder_dim]
    am_pruned = torch.rand(2, 100, 5, 512)
    lm_pruned = torch.rand(2, 100, 5, 512)
    res = attn(am_pruned, lm_pruned)
    print("__main__ res", res.shape)
