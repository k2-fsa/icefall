#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao)
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

import copy
import logging
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
from encoder_interface import EncoderInterface
from zapformer_modules import (
    ActivationAndLinear,
    CausalSequenceNorm,
    CorrelationLimiter,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    OrthogonalLinear,
    RmsNorm,
    SequenceNorm,
    ScaledLinear,  # just an initializer for Linear
    SwashR,
    ScaleLimiter,
)
from zapformer_utils import (
    limit_param_value,
    penalize_abs_values_gt,
    softmax,
    with_loss,
)


from torch import Tensor, nn

from icefall.utils import make_pad_mask


class Zapformer(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        pos_head_dim (int or Tuple[int]): dimension of position-embedding in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_multiple (int or Tuple[int]): determines hidden dimension in feedforward modules
        conv_params (int or Tuple[int])): Kernel size of convolution module

        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.
    """
    def __init__(
        self,
        input_dim: int,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        query_head_dim: Union[int, Tuple[int]] = 64,
        value_head_dim: Union[int, Tuple[int]] = 12,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_multiple: Union[int, Tuple[int]] = 4,
        conv_params: Union[int, Tuple[int]] = 31,
        num_freqs: int = 64,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super(Zapformer, self).__init__()

        def _to_tuple(x):
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor  # int

        self.downsampling_factor = downsampling_factor  # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        self.pos_head_dim = pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_multiple = _to_tuple(feedforward_multiple)
        self.conv_params = conv_params = _to_tuple(conv_params)
        self.num_freqs = num_freqs

        self.causal = causal
        self.chunk_size = (chunk_size,) if isinstance(chunk_size, int) else chunk_size
        self.left_context_frames = (left_context_frames,) if isinstance(left_context_frames, int) else left_context_frames

        # each one will be ZapformerEncoder or OrthogonalDownsample or OrthogonalUpsample
        encoders = []

        num_encoders = len(downsampling_factor)

        # caution: some changes we made for this break the streaming, later we'll try to fix this.
        encoders_downsampling_factors = [ ]

        # make it so large the limit is never reached.
        max_proj_dim = max(downsampling_factor) * max(encoder_dim)


        for i in range(num_encoders):
            encoder_layer = ZapformerEncoderLayer(
                embed_dim=encoder_dim[i],
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                value_head_dim=value_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                feedforward_multiple=feedforward_multiple[i],
                conv_params=conv_params[i],
                num_freqs=num_freqs,
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = ZapformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                dim=downsampling_factor[i]*input_dim,
            )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # Share a single AngularFreqBasis instance across all layers within each encoder stack
        for encoder in self.encoders:
            shared_basis = encoder.layers[0].self_attn.rel_pos.angular_freq_basis
            for layer in encoder.layers[1:]:
                layer.self_attn.rel_pos.angular_freq_basis = shared_basis

        self.out_norm = RmsNorm()


    def get_chunk_info(self) -> Tuple[int, int]:
        """
         Returns chunk_size and left_context_chunks.
         """
        if not self.causal:
            return -1, -1

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert len(self.chunk_size) == 1, self.chunk_size
            chunk_size = self.chunk_size[0]
        else:
            chunk_size = random.choice(self.chunk_size)

        if chunk_size == -1:
            left_context_chunks = -1
        else:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                assert len(self.left_context_frames) == 1, self.left_context_frames
                left_context_frames = self.left_context_frames[0]
            else:
                left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1

        return chunk_size, left_context_chunks

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
          aux_loss_scale:
            If supplied, auxiliary losses such as CosineSimilarityLoss will be
            applied with this scale on the loss (note, these aux losses are
            reduced via summation over frames.)
        Returns:
          Return (embeddings_lengths), where:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        chunk_size, left_context_chunks = self.get_chunk_info()
        orig_seq_len = x.shape[0]

        pad = (-orig_seq_len) % max(self.downsampling_factor)
        # pad sequence length to be multiple of max(self.downsampling_factor)
        x = torch.cat((x, x[-1:].repeat(pad, 1, 1)),
                      dim=0)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # Not support exporting a model for simulating streaming decoding
            attn_mask = None
        else:
            attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        src_key_padding_mask = pad_mask(src_key_padding_mask, x.shape[0])

        num_stacks = len(self.downsampling_factor)


        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = downsample_by(x, ds)
            T = x.shape[0]
            x = module(
                x,
                chunk_size=chunk_size // ds if chunk_size > 0 else -1,
                src_key_padding_mask=(
                    None
                    if src_key_padding_mask is None
                    else src_key_padding_mask[..., ::ds]
                ),
                attn_mask=(None
                           if attn_mask is None
                           else attn_mask[::ds, ::ds]
                ),
                aux_loss_scale=aux_loss_scale * ds / (self.output_downsampling_factor * num_stacks)
            )
            x = upsample_by(x, ds)

        od = self.output_downsampling_factor
        x = downsample_by(x, od)
        x = x[:(orig_seq_len + od - 1) // od]  # truncate so seq len not affected by padding

        if od > 1:
            x_lens = (x_lens + od - 1) // od

        x = self.out_norm(x)

        # disable the projection-overlap loss
        #if self.training:
        #    # all of our losses and aux losses are proportional to the number of frames of data, so
        #    # we multiply by that factor.
        #    x = with_loss(x, aux_loss_scale * x.shape[0] * x.shape[1] * self.compute_projection_overlap())

        return x, x_lens


    def compute_projection_overlap(self, verbose: bool = False):
        # This is currently just used for some diagnostics.

        # It also computes an auxiliary loss (currently unused) that
        # ensures that the projections from more-subsampled sequences "contain" enough of the
        # projections from the less-subsampled sequences-- specifically the direction where
        # all the less-subsampled projections co-vary in the same way, e.g. if there are
        # two frames, that the two frames are identical.

        min_overlap = 0.6  # we can tune this.  CAUTION: I turned off this aux loss by commenting
        # it out in forward(),

        tot_loss = 0.0
        # between pairs of encoders
        N = len(self.encoders)

        covs = []
        ranks = []
        for i in range(N):
            proj_i = self.encoders[i].proj.get_weight()
            cov_i = torch.matmul(proj_i.t(), proj_i)
            covs.append(cov_i)
            ranks.append(proj_i.shape[0])

        for i in range(N):
            for j in range(i):
                cov_i, cov_j = covs[i], covs[j]
                rank_i, rank_j = ranks[i], ranks[j]
                if cov_i.shape[0] > cov_j.shape[0]:
                    cov_i, cov_j = cov_j, cov_i
                    rank_i, rank_j = rank_j, rank_i
                dim_i = cov_i.shape[0]  # now this is <= proj_j.shape[0]
                dim_j = cov_j.shape[0]
                assert dim_i <= dim_j
                assert dim_j % dim_i == 0  # dims must be multiples of each other   (these are the
                # feature dimension prior to project, i.e. the larger dimensions.)
                R = dim_j // dim_i  # e.g. 1, 2, 4
                assert R in [1, 2, 4, 8, 16]
                cov_i = cov_i.repeat(R, R) * (1. / R)
                # denominator is the minimum of the two ranks,
                # because due to the orthogonal constraint, the maximum possible value of (cov_i * cov_j).sum() would be the
                # smaller of the two ranks.
                cosine = (cov_i * cov_j).sum() / min(rank_i, rank_j)

                loss = (min_overlap - cosine).relu()
                tot_loss = tot_loss + loss
                if verbose:
                    logging.info(f"overlap[{i}, {j}] = {cosine}")
        return tot_loss


    def _get_attn_mask(
        self, x: Tensor, chunk_size: int, left_context_chunks: int
    ) -> Optional[Tensor]:
        """
        Return None if chunk_size == -1, else return attention mask of shape
          (seq_len, seq_len), interpreted as (tgt_seq_len, src_seq_len).  True
           means a masked position.
        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
          chunk_size: chunk size, must divide
        """
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all(
                chunk_size * left_context_chunks
                >= (self.conv_params[i] // 2) * self.downsampling_factor[i]
                for i in range(num_encoders)
            )  # TODO: could test remove this
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            c = t // chunk_size
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c, src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        caches: List[Tensor],
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          caches: list of cached tensors of all encoder layers. For layer-i,
            caches[i*9:(i+1)*9] is (cached_key, cached_value, cached_conv,
            cached_norm_stats, cached_norm_len, cached_attn_wm_sum,
            cached_attn_wm_num_frames, cached_conv_wm_sum, cached_conv_wm_num_frames).
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.

        Returns:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
            - updated caches: an updated list of cache tensors.
        """
        orig_seq_len = x.shape[0]
        pad = (-orig_seq_len) % max(self.downsampling_factor)
        # pad sequence length to be multiple of max(self.downsampling_factor)
        x = torch.cat((x, x[-1:].repeat(pad, 1, 1)), dim=0)

        if src_key_padding_mask is not None:
            left_context_frames = src_key_padding_mask.shape[1] - orig_seq_len
            assert left_context_frames == self.left_context_frames[0]
            if pad > 0:
                src_key_padding_mask = torch.cat(
                    (src_key_padding_mask[:, :left_context_frames],
                    pad_mask(src_key_padding_mask[:, left_context_frames:], x.shape[0])),
                    dim=1,
                )

        new_caches = []
        layer_offset = 0

        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            ds = self.downsampling_factor[i]

            x = downsample_by(x, ds)

            # Slice out the specific caches for the current module
            module_caches = caches[layer_offset * 9 : (layer_offset + num_layers) * 9]

            x, new_module_caches = module.streaming_forward(
                src=x,
                caches=module_caches,
                left_context_len=self.left_context_frames[0] // ds,
                src_key_padding_mask=(
                    None
                    if src_key_padding_mask is None
                    else src_key_padding_mask[..., ::ds]
                ),
            )

            layer_offset += num_layers
            new_caches.extend(new_module_caches)

            x = upsample_by(x, ds)

        # Output downsampling and normalization
        od = self.output_downsampling_factor
        x = downsample_by(x, od)

        x = x[:(orig_seq_len + od - 1) // od]  # truncate so seq len not affected by padding

        if od > 1:
            x_lens = (x_lens + od - 1) // od

        x = self.out_norm(x)

        return x, x_lens, new_caches

    @torch.jit.export
    def get_init_caches(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Get initial caches.

        A list of cached tensors of all encoder layers. For layer-i, caches[i*9:(i+1)*9]
        is (cached_key, cached_value, cached_conv, cached_norm_stats, cached_norm_len,
        cached_attn_wm_sum, cached_attn_wm_num_frames, cached_conv_wm_sum, cached_conv_wm_num_frames).
        """
        caches = []
        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            embed_dim = self.encoder_dim[i]
            ds = self.downsampling_factor[i]
            num_heads = self.num_heads[i]
            key_dim = self.query_head_dim[i] * num_heads
            value_dim = self.value_head_dim[i] * num_heads
            downsample_left = self.left_context_frames[0] // ds
            conv_left_pad = self.conv_params[i] - 1

            for layer_idx, enc_layer in enumerate(module.layers):
                cached_key = torch.zeros(downsample_left, batch_size, key_dim, device=device)
                cached_value = torch.zeros(downsample_left, batch_size, value_dim, device=device)
                cached_conv = torch.zeros(batch_size, embed_dim, conv_left_pad, device=device)
                cached_norm_stats, cached_norm_len = enc_layer.norm.get_init_cache(batch_size)
                cached_norm_stats = cached_norm_stats.to(device)
                cached_norm_len = cached_norm_len.to(device)

                attn_value_dim = self.value_head_dim[i] * num_heads
                cached_attn_wm_sum = torch.zeros(1, batch_size, attn_value_dim, device=device)
                cached_attn_wm_num_frames = torch.zeros(batch_size, dtype=torch.int64, device=device)
                cached_conv_wm_sum = torch.zeros(1, batch_size, embed_dim, device=device)
                cached_conv_wm_num_frames = torch.zeros(batch_size, dtype=torch.int64, device=device)

                caches.extend([
                    cached_key,
                    cached_value,
                    cached_conv,
                    cached_norm_stats,
                    cached_norm_len,
                    cached_attn_wm_sum,
                    cached_attn_wm_num_frames,
                    cached_conv_wm_sum,
                    cached_conv_wm_num_frames,
                ])

        return caches


def pad_mask(mask: Optional[Tensor], seq_len: int):
    # mask: (batch_size, old_seq_len)
    # if mask is not None, returns mask: (batch_size, seq_len); pads with True (i.e., masked).
    if mask is None:
        return None
    (batch_size, old_seq_len) = mask.shape
    pad = seq_len - old_seq_len
    if pad == 0:
        return mask
    else:
        return torch.cat((mask, torch.ones(batch_size, pad, device=mask.device, dtype=torch.bool)),
                         dim=1)


def downsample_by(x: Tensor, downsampling_factor: int) -> Tensor:
    # x: (seq_len, batch_size, num_channels)
    # Returns: (seq_len // downsampling_factor, batch_size, num_channels * downsampling_factor)
    if downsampling_factor == 1:
        return x
    (seq_len, batch_size, num_channels) = x.shape
    x = x.reshape(seq_len // downsampling_factor, downsampling_factor, batch_size, num_channels)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(seq_len // downsampling_factor, batch_size, downsampling_factor * num_channels)
    return x

def upsample_by(x: Tensor, upsampling_factor: int) -> Tensor:
    # x: (seq_len, batch_size, num_channels)
    # Returns: (seq_len * upsampling_factor, batch_size, num_channels // upsampling_factor)
    if upsampling_factor == 1:
        return x
    (seq_len, batch_size, num_channels) = x.shape
    x = x.reshape(seq_len, batch_size, upsampling_factor, num_channels // upsampling_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(seq_len * upsampling_factor, batch_size, num_channels // upsampling_factor)
    return x


def get_dct_matrix(N):
    """
    Generates an orthonormal DCT-II matrix for a given size N.
    Args:
        N (int): The size of the square matrix.
    Returns:
        torch.Tensor: The N x N orthonormal DCT-II matrix.
    """
    # Create the base matrix with dimensions (N, N)
    mat = torch.zeros(N, N)
    # Create a tensor for the indices k (rows) and n (columns)
    k = torch.arange(N).unsqueeze(1)
    n = torch.arange(N).unsqueeze(0)
    # Fill the matrix using the DCT-II formula
    mat = math.sqrt(2 / N) * torch.cos(math.pi / (2 * N) * (2 * n + 1) * k)
    # Adjust the first row (k=0) with a special normalization factor
    mat[0] *= (2 ** -0.5)
    return mat


class ZapformerEncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_multiple: determines the hidden dimension of the feedforward module

        conv_params (int): params per channel of convolution module

    Examples::
        >>> encoder_layer = ZapformerEncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query_head_dim: int,
        value_head_dim: int,
        pos_head_dim: int,
        feedforward_multiple: int,
        conv_params: int,
        num_freqs: int = 64,
        causal: bool = False,
    ) -> None:
        super(ZapformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.name = None  # will be set from training loop

        self.offset_scale_limiter = ScaleLimiter(max_rms=1.0)

        #power = 0.35  # power should be between 0 and 1.  1 would mean cov == I (unattainable)
        #limit = (1. / (embed_dim  ** power)))
        limit = 0.25  # this is very enormous limit on correlations, it's just to prevent divergence
        # and bad parameter locations from which it's impossible for the optimizer to escape.  i.e.
        # it should impose no real limitation on "normal" training runs.
        self.correlation_limiter = CorrelationLimiter(limit=limit)

        self.self_attn = MultiheadRelPosGatedSelfAttention(
            embed_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            value_head_dim=value_head_dim,
            pos_head_dim=pos_head_dim,
            num_freqs=num_freqs,
            causal=causal,
        )

        feedforward_dim = embed_dim * feedforward_multiple
        self.feed_forward1 = FeedforwardModule(embed_dim, feedforward_dim)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim)

        self.conv_module = ConvolutionModule(embed_dim, conv_params, causal=causal)

        self.norm = CausalSequenceNorm() if causal else SequenceNorm()

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tensor:
        """
            Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
             chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
             attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                    interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                   True means masked position. May be None.
        src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.
          aux_loss_scale:
            If supplied, auxiliary losses such as CosineSimilarityLoss will be
            applied with this scale on the loss (note, these aux losses are
            reduced via summation over frames.)

            Returns:
               A tensor which has the same shape as src
        """
        src_orig = src

        src = with_loss(src, self.correlation_limiter(src.permute(1, 0, 2),
                                                      2. * aux_loss_scale, mask=src_key_padding_mask),
                        None)

        src_pre_ff1 = src

        src = src + self.feed_forward1(src, aux_loss_scale=0.1 * aux_loss_scale, src_key_padding_mask=src_key_padding_mask)

        # may try changing src_pre_ff1 to src or vice versa.
        src = src + self.self_attn(src_pre_ff1, src, attn_mask=attn_mask,
                                   key_padding_mask=src_key_padding_mask,
                                   aux_loss_scale=0.1 * aux_loss_scale)

        src = src + self.conv_module(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask, aux_loss_scale=0.1 * aux_loss_scale)

        src = src + self.feed_forward2(src, aux_loss_scale=0.1 * aux_loss_scale, src_key_padding_mask=src_key_padding_mask)

        residual_scale = 0.25
        offset = (src - src_orig) * residual_scale

        offset = self.offset_scale_limiter(offset, aux_loss_scale)

        src = src_orig + offset

        src = self.norm(src, src_key_padding_mask)

        src = src.clamp(min=-5, max=5)

        return src

    def streaming_forward(
        self,
        src: Tensor,
        cached_key: Tensor,
        cached_value: Tensor,
        cached_conv: Tensor,
        cached_norm_stats: Tensor,
        cached_norm_len: Tensor,
        cached_attn_wm_sum: Tensor,
        cached_attn_wm_num_frames: Tensor,
        cached_conv_wm_sum: Tensor,
        cached_conv_wm_num_frames: Tensor,
        left_context_len: int,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pass the input through the encoder layer in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            cached_key: cached attention key tensor, of shape (left_context_len, batch_size, key_dim)
            cached_value: cached attention value tensor, of shape (left_context_len, batch_size, value_dim)
            cached_conv: cached left context for the convolution module, of shape (batch_size, channels, left_pad)
            cached_norm_stats: cached SequenceNorm stats, of shape (batch_size,)
            cached_norm_len: cached SequenceNorm length, scalar.
            cached_attn_wm_sum: (1, batch, channels), cumulative sum for attention weighted_mean
            cached_attn_wm_num_frames: (batch,), number of frames for attention weighted_mean
            cached_conv_wm_sum: (1, batch, channels), cumulative sum for conv weighted_mean
            cached_conv_wm_num_frames: (batch,), number of frames for conv weighted_mean
            left_context_len: number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, left_context_len + seq_len);
                True means masked position. May be None.

        Returns:
            - x, with the same shape as src
            - updated cached_key
            - updated cached_value
            - updated cached_conv
            - updated cached_norm_stats
            - updated cached_norm_len
            - updated cached_attn_wm_sum
            - updated cached_attn_wm_num_frames
            - updated cached_conv_wm_sum
            - updated cached_conv_wm_num_frames
        """
        src_orig = src

        src_pre_ff1 = src

        chunk_mask = None if src_key_padding_mask is None else src_key_padding_mask[:, left_context_len:]

        src = src + self.feed_forward1(src, src_key_padding_mask=chunk_mask)

        # may try changing src_pre_ff1 to src or vice versa.
        self_attn_out, cached_key, cached_value, cached_attn_wm_sum, cached_attn_wm_num_frames = self.self_attn.streaming_forward(
            x_qkp=src_pre_ff1,
            x_vg=src,
            left_context_len=left_context_len,
            cached_key=cached_key,
            cached_value=cached_value,
            cached_wm_sum=cached_attn_wm_sum,
            cached_wm_num_frames=cached_attn_wm_num_frames,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self_attn_out

        src_conv, cached_conv, cached_conv_wm_sum, cached_conv_wm_num_frames = self.conv_module.streaming_forward(
            src,
            cached_conv=cached_conv,
            cached_wm_sum=cached_conv_wm_sum,
            cached_wm_num_frames=cached_conv_wm_num_frames,
            src_key_padding_mask=chunk_mask,
        )
        src = src + src_conv

        src = src + self.feed_forward2(src, src_key_padding_mask=chunk_mask)

        residual_scale = 0.25
        offset = (src - src_orig) * residual_scale

        src = src_orig + offset

        src, cached_norm_stats, cached_norm_len = self.norm.streaming_forward(
            src,
            cached_stats_sum=cached_norm_stats,
            cached_len=cached_norm_len,
        )

        src = src.clamp(min=-5, max=5)
        
        return (
            src,
            cached_key,
            cached_value,
            cached_conv,
            cached_norm_stats,
            cached_norm_len,
            cached_attn_wm_sum,
            cached_attn_wm_num_frames,
            cached_conv_wm_sum,
            cached_conv_wm_num_frames,
        )


class ZapformerEncoder(nn.Module):
    r"""ZapformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ZapformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
         dim:  the dimension of the input and output (layer dim may be less than this).

    Examples::
        >>> encoder_layer = ZapformerEncoderLayer(embed_dim=512, nhead=8)
        >>> zapformer_encoder = ZapformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zapformer_encoder(src)
    """
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dim: int,
    ) -> None:
        super().__init__()

        # self.downsample will also reverse the downsampling operation for us afterward.
        self.proj = OrthogonalLinear(dim,
                                     encoder_layer.embed_dim,
                                     bias=False)

        self.name = None
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        self.residual_scales = nn.Parameter(
            torch.cat([ -1.0 * torch.ones(1),
                        (1. / num_layers) * torch.ones(num_layers) ],
                      dim=0))

        self.input_scale = nn.Parameter(torch.tensor([1.0]))

        self.copy_bypass = Identity()

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embed_dim),
                 but embed_dim is allowed to exceed the modules' embed_dim; we will bypass
                 any extra dimensions.
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns:
             (out, out_sd), both of the same shape as src,
           where out_sd is an alternative version of out for stochastic-depth, that does not see the bypass.
        """
        src_orig_fulldim = src

        src = self.proj(src)  # project to layer dim.

        num_layers = len(self.layers)
        src_orig = src

        residual_scale = limit_param_value(self.residual_scales[0],
                                           min=-1.0, max=-0.5)
        input_scale = limit_param_value(self.input_scale,
                                        min=0.5, max=2.0)

        src_with_bypass = residual_scale * src
        src = input_scale * src

        for i, mod in enumerate(self.layers):

            src = mod(
                src,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
                aux_loss_scale=aux_loss_scale/num_layers,
            )
            residual_scale = limit_param_value(self.residual_scales[i + 1],
                                               min=0.0 if i + 1 < num_layers else min(0.5, 1. / num_layers),
                                               max=1.0)
            src_with_bypass = src_with_bypass + residual_scale * src


        offset = src_with_bypass

        src = src_orig_fulldim + self.proj(offset, transpose=True)
        # in effect src_orig_fulldim already contains src_orig with a scale of 1 for the missing dims,
        # because of some identities involving orthogonal matrices.

        return src


    def streaming_forward(
        self,
        src: Tensor,
        caches: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Pass the input through the encoder layers in turn in streaming mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embed_dim).
            caches: list of cached tensors of N encoder layers. For layer-i,
              caches[i*9:(i+1)*9] is (cached_key, cached_value, cached_conv,
              cached_norm_stats, cached_norm_len, cached_attn_wm_sum,
              cached_attn_wm_num_frames, cached_conv_wm_sum, cached_conv_wm_num_frames).
            left_context_len: Number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
          - output, a Tensor with the same shape as src.
          - updated caches
        """
        src_orig_fulldim = src

        # project to layer dim.
        src = self.proj(src)

        num_layers = len(self.layers)
        assert len(caches) == num_layers * 9

        residual_scale = self.residual_scales[0]
        input_scale = self.input_scale

        src_with_bypass = residual_scale * src
        src = input_scale * src

        new_caches = []
        for i, mod in enumerate(self.layers):
            (
                cached_key,
                cached_value,
                cached_conv,
                cached_norm_stats,
                cached_norm_len,
                cached_attn_wm_sum,
                cached_attn_wm_num_frames,
                cached_conv_wm_sum,
                cached_conv_wm_num_frames,
            ) = caches[i * 9 : (i + 1) * 9]

            (
                src,
                new_cached_key,
                new_cached_value,
                new_cached_conv,
                new_cached_norm_stats,
                new_cached_norm_len,
                new_cached_attn_wm_sum,
                new_cached_attn_wm_num_frames,
                new_cached_conv_wm_sum,
                new_cached_conv_wm_num_frames,
            ) = mod.streaming_forward(
                src,
                cached_key=cached_key,
                cached_value=cached_value,
                cached_conv=cached_conv,
                cached_norm_stats=cached_norm_stats,
                cached_norm_len=cached_norm_len,
                cached_attn_wm_sum=cached_attn_wm_sum,
                cached_attn_wm_num_frames=cached_attn_wm_num_frames,
                cached_conv_wm_sum=cached_conv_wm_sum,
                cached_conv_wm_num_frames=cached_conv_wm_num_frames,
                left_context_len=left_context_len,
                src_key_padding_mask=src_key_padding_mask,
            )

            layer_residual_scale = self.residual_scales[i + 1]

            src_with_bypass = src_with_bypass + layer_residual_scale * src

            new_caches.extend([
                new_cached_key,
                new_cached_value,
                new_cached_conv,
                new_cached_norm_stats,
                new_cached_norm_len,
                new_cached_attn_wm_sum,
                new_cached_attn_wm_num_frames,
                new_cached_conv_wm_sum,
                new_cached_conv_wm_num_frames,
            ])

        offset = src_with_bypass
        src = src_orig_fulldim + self.proj(offset, transpose=True)

        return src, new_caches




class MultiheadRelPosGatedSelfAttention(nn.Module):
    r"""
    Module that computes multi-head attention weights with additive relative-position
    scores that are kept separate from the regular scores.  The values have gating.
    An RMSNorm module is used to pre-normalize the input embedding only as it is
    input to the queries and keys, not the values.

    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int ,
        value_head_dim: int,
        num_freqs: int = 64,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.name = None  # will be overwritten in training code; for diagnostics.

        self.in_norm = RmsNorm()

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.qkp_in_proj = ScaledLinear(
            embed_dim, in_proj_dim,
            bias=True, initial_scale=0.125,
        )

        self.rel_pos = RelPosScores(num_heads, pos_head_dim, num_freqs=num_freqs)

        self.copy_query = Identity()
        self.copy_pos_query = Identity()

        # value and gating in_proj.
        self.vg_in_proj = ScaledLinear(embed_dim, 3 * num_heads * value_head_dim,
                                       initial_scale=0.1, bias=True)

        self.copy_v = nn.Identity()  # diagnostics.
        self.sigmoid_in = nn.Sigmoid()
        self.sigmoid_out = nn.Sigmoid()

        # out proj for the value times gating.
        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.1
        )

        self.weighted_mean = WeightedMean(num_heads * value_head_dim, causal) # TODO: fix causal option

    def forward(
        self,
        x_qkp: Tensor,
        x_vg: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        aux_loss_scale: float = 0.0,
    ) -> Tensor:
        r"""
        Args:
            x_qkp: input of shape (seq_len, batch_size, embed_dim), that is used for the queries,
                  keys and positions.
            x_vg: input of shape (seq_len, batch_size, embed_dim), that is used for the values
                  and gates.  May be the same as x_qk.
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        query_head_dim = self.query_head_dim
        num_heads = self.num_heads
        x_qkp = self.in_norm(x_qkp)
        x_qkp = self.qkp_in_proj(x_qkp)

        seq_len, batch_size, _ = x_qkp.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x_qkp[..., 0:query_dim]
        k = x_qkp[..., query_dim : 2 * query_dim] * (query_head_dim ** -0.5)
        p = x_qkp[..., 2 * query_dim:]

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, -1)

        #q = self.rope(q.permute(1, 0, 2, 3))  # (batch, seq, head, channel)
        #k = self.rope(k.permute(1, 0, 2, 3))  # (batch, seq, head, channel)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)  # (head, batch, time1, time2)

        p = p.permute(1, 2, 0, 3)
        pos_scores = self.rel_pos(p)  # (batch, head, time1, time2)
        attn_scores = attn_scores + pos_scores.permute(1, 0, 2, 3)


        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size,
                seq_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and self.training:
            attn_scores_limit = 8.0  # limit on our metric that affects how much grad we are likely to backpropagate.
            attn_scores = PenalizeLargeAttentionScores.apply(attn_scores, attn_scores_limit,
                                                             0.1 * aux_loss_scale,
                                                             key_padding_mask, self.name)

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        vg = self.vg_in_proj(x_vg)
        N = vg.shape[-1] // 3
        v = vg[..., :N]
        g = vg[..., N:]
        if self.training:
            # don't let the sigmoid values get too extreme, limit to -2..2.
            g = penalize_abs_values_gt(g, 2, penalty=0.02*aux_loss_scale)

        g_in, g_out = g.chunk(2, dim=-1)
        v = v * self.sigmoid_in(g_in)

        wm = self.weighted_mean(v, key_padding_mask, apply_mask=True)

        v = v.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        v = self.copy_v(v)
        value_head_dim = v.shape[-1]
        # now v: (num_heads, batch_size, seq_len, value_head_dim)

        # todo: see whether there is benefit in overriding matmul
        v = torch.matmul(attn_weights, v)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        v = (
            v.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )


        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        v = v + wm
        v = v * self.sigmoid_out(g_out)
        v = self.out_proj(v)
        return v

    def streaming_forward(
        self,
        x_qkp: Tensor,
        x_vg: Tensor,
        left_context_len: int,
        cached_key: Tensor,
        cached_value: Tensor,
        cached_wm_sum: Tensor,
        cached_wm_num_frames: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Args:
            x_qkp: input of shape (seq_len, batch_size, embed_dim), that is used for the queries,
                  keys and positions.
            x_vg: input of shape (seq_len, batch_size, embed_dim), that is used for the values
                  and gates.  May be the same as x_qk.
            left_context_len: length of the cached left context.
            cached_key: cached attention key tensor, of shape (left_context_len, batch_size, key_dim).
            cached_value: cached attention value tensor, of shape (left_context_len, batch_size, value_dim).
            cached_wm_sum: (1, batch, channels), cumulative sum for weighted_mean
            cached_wm_num_frames: (batch,), number of frames seen so far
            key_padding_mask: a bool tensor of shape (batch_size, left_context_len + seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.

        Returns:
            - attention output, of shape (seq_len, batch_size, embed_dim)
            - updated cached_key, of shape (left_context_len, batch_size, key_dim)
            - updated cached_value, of shape (left_context_len, batch_size, value_dim)
            - Updated cached_wm_sum (1, batch, channels)
            - Updated cached_wm_num_frames (batch,)
        """
        query_head_dim = self.query_head_dim
        num_heads = self.num_heads
        x_qkp = self.in_norm(x_qkp)
        x_qkp = self.qkp_in_proj(x_qkp)

        seq_len, batch_size, _ = x_qkp.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x_qkp[..., 0:query_dim]
        k = x_qkp[..., query_dim : 2 * query_dim]
        p = x_qkp[..., 2 * query_dim:]

        # append the cached key to the current key, and update the cache
        assert cached_key.shape[0] == left_context_len, (cached_key.shape, left_context_len)
        k = torch.cat([cached_key, k], dim=0)
        kv_len = k.shape[0]
        cached_key = k[kv_len - left_context_len:]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        k = k.reshape(kv_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, -1)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, query_head_dim, time2)

        attn_scores = torch.matmul(q, k)  # (head, batch, time1, time2)

        p = p.permute(1, 2, 0, 3)
        pos_scores = self.rel_pos(p, left_context_len)  # (batch, head, time1, time2)
        attn_scores = attn_scores + pos_scores.permute(1, 0, 2, 3)

        assert attn_scores.shape == (num_heads, batch_size, seq_len, kv_len)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, kv_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1000)

        attn_weights = attn_scores.softmax(dim=-1)

        vg = self.vg_in_proj(x_vg)
        N = vg.shape[-1] // 3
        v = vg[..., :N]
        g = vg[..., N:]
        g_in, g_out = g.chunk(2, dim=-1)
        v = v * self.sigmoid_in(g_in)

        wm, cached_wm_sum, cached_wm_num_frames = self.weighted_mean.streaming_forward(
            v, cached_wm_sum, cached_wm_num_frames
        )

        # append the cached value to the current value, and update the cache
        assert cached_value.shape[0] == left_context_len, (cached_value.shape, left_context_len)
        v = torch.cat([cached_value, v], dim=0)
        cached_value = v[kv_len - left_context_len:]

        v = v.reshape(kv_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        value_head_dim = v.shape[-1]
        # now v: (num_heads, batch_size, kv_len, value_head_dim)

        # todo: see whether there is benefit in overriding matmul
        v = torch.matmul(attn_weights, v)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        v = (
            v.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        v = v + wm
        v = v * self.sigmoid_out(g_out)
        v = self.out_proj(v)

        return v, cached_key, cached_value, cached_wm_sum, cached_wm_num_frames

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
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


class PenalizeLargeAttentionScores(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            attn_scores: Tensor,
            limit: float,
            aux_loss_scale: float,
            key_padding_mask: Optional[Tensor],
            name: str):
        # attn_scores: (head, batch, query_time, key_time)
        ctx.save_for_backward(attn_scores)
        ctx.mask = key_padding_mask    # has no grad
        ctx.limit = limit
        ctx.aux_loss_scale = aux_loss_scale
        ctx.name = name
        return attn_scores

    @staticmethod
    def backward(
            ctx,
            attn_scores_grad):
        attn_scores, = ctx.saved_tensors
        mask = ctx.mask
        (num_heads, batch_size, seq_len, _) = attn_scores.shape
        with torch.amp.autocast('cuda', enabled=False):
            attn_scores = attn_scores.to(torch.float)
            attn_scores = attn_scores.detach()
            # attn_scores: (head, batch, query_time, key_time)
            attn_scores.requires_grad = True
            with torch.enable_grad():
                probs = attn_scores.softmax(dim=-1)
                scaled_scores = attn_scores.abs() * probs
                avg_scores = scaled_scores.sum(dim=-1) # (head, batch, query_time)
                if mask is not None:
                    avg_scores = avg_scores * (~mask) # mask: (batch, time)
                query_scores = (avg_scores - ctx.limit).relu()

                if random.random() < 0.0005:
                    query_excess = query_scores.mean(dim=(1,2)).to('cpu')
                    avg_scores_mean = avg_scores.mean(dim=(1,2)).to('cpu')
                    logging.info(f"PenalizeLargeAttentionScores: {ctx.name}, limit={ctx.limit}, avg_scores={avg_scores_mean}, query_excess={query_excess}")
                # all these losses have a "per-frame" scaling, i.e. scaled proportional to the total number
                # of frames which is batch_size * seq_len.  normalize by dividing by num heads.
                # also divide by ctx.limit so it's like penalizing a relative excess.
                query_scores.backward(gradient=torch.full_like(query_scores, ctx.aux_loss_scale / (num_heads * ctx.limit)))

        return attn_scores_grad + attn_scores.grad, None, None, None, None





class FeedforwardModule(nn.Module):
    """Feedforward module in Zapformer model."""

    def __init__(self, embed_dim: int, feedforward_dim: int):
        super(FeedforwardModule, self).__init__()
        # try to get in the useful range of the activation function, i.e. not too small.
        self.in_proj = ScaledLinear(embed_dim, feedforward_dim)
        # weight_min_rms will be interpreted by get_parameter_groups_with_lrs() and passed
        # to the TransformedAdam optimizer.
        self.in_proj.weight_min_rms = 0.02

        self.out_proj = ActivationAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwashR",
            initial_scale=0.5,
            bias=True,
        )


    def forward(self, x: Tensor, aux_loss_scale: float = 0.0, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.in_proj(x)
        x = self.out_proj(x)
        return x

def round_up_to_power_of_two(x):
    x = x - 1
    x = x | x >> 1
    x = x | x >> 2
    x = x | x >> 4
    x = x | x >> 8
    x = x | x >> 16
    x = x + 1
    return x


# wolfram alpha:
# the right part of the triangular bin, from 0 to +W.
# definite integral from omega = 0 to W of (1 - omega/W) exp(-i x \omega) d\omega
# = -(i W x + e^(-i W x) - 1)/(W x^2)
# Re[definite integral from omega = 0 to W of (1 - omega/W) exp(-i x \omega) d\omega]
# = (1 - cos(W x))/(W x^2)
# Im[definite integral from omega = 0 to W of (1 - omega/W) exp(-i x \omega) d\omega]
# = (sin(W x) - W x)/(W x^2)

# the left part of the triangular bin, from -W to 0.
# definite integral from omega = -W to 0 of (omega/W + 1) exp(-i x \omega) d\omega
# (i W x - e^(i W x) + 1)/(W x^2)
#
# Let the center frequency be C.
# right side:
# =   e^(i C x) * -(i W x + e^(-i W x) - 1)/(W x^2)
# "alternate form including W, C and x are real":  [note, this is left hand width, W_l]
# (W x sin(C x) - cos(x (C - W)) + cos(C x))/(W x^2) - (i (sin(x (C - W)) + W x cos(C x) - sin(C x)))/(W x^2)
#
# left side:
# e^(i C x) *  (i W x - e^(i W x) + 1)/(W x^2)
# "alternate form including W, C and x are real": [note, this is right hand width, W_r]
# -(W x sin(C x) + cos(x (C + W)) - cos(C x))/(W x^2) + (i (-sin(x (C + W)) + W x cos(C x) + sin(C x)))/(W x^2)
#
# summing the left and right sides:
# Real part:
#
#   (W_r x sin(C x) - cos(x (C - W_r)) + cos(C x))/(W_r x^2)
#   -(W_l x sin(C x) + cos(x (C + W_l)) - cos(C x))/(W_l x^2)
# =   (cos(C x) - cos((C - W_r)x))   / W_r x^2
#   + (cos(C x) - cos((C + W_l)x))   / W_l x^2

# Imaginary part:
#  -(sin(x (C - W_r)) + W_r x cos(C x) - sin(C x))) / (W_r x^2)
#  +(-sin(x (C + W_l)) + W_l x cos(C x) + sin(C x)) / (W_l x^2)
#  =  ( sin(C x) - sin((C - W_r)x) ) / (W_r x^2)
#   + ( sin(C x) - sin((C + W_l)x) ) / (W_l x^2)

def compute_angular_freq_basis_triangular(freqs: Tensor,
                                          t: Tensor,
                                          scale: bool) -> Tensor:
    """
    This function computes a set of windowed sinusoidal functions
    corresponding to the real and imaginary parts  of possibly-asymmetrical
    triangular angular-frequency bins in frequency space.  This basis
    allows you to approximate functions whose fourier spectrum is
    a piecewise linear function of frequency, with the x-axis values of
    the inflection points of the piecewise linear function corresponding
    to the supplied "freqs".

  Args:
     freqs: the frequencies of the triangular-bin centers; the left and
         right parts of the widths of the triangular bins correspond to the
         distances to the two adjacent bins; for the "edge" bins, the
         "edge" distances are duplicated.
      t: the "t" (or x) values for which we want to evaluate the basis; this
        will normally be some kind of arange expression e.g. arange(100).
   scale: if True, the returned basis will contain the "natural" scaling
        factors that arise from the bin widths; if False, it will be
        normalized so that the maximum absolute value of the real
        functions (attained at t==0) is 1.


   Returns:
        Returns the real and imaginary parts of the basis functions, with
        shape (t.size(), freqs.size(), 2)
    """
    dtype = freqs.dtype
    freqs = freqs.to(torch.double)
    t = t.to(torch.double)

    t = t.unsqueeze(-1)


    C = freqs  # Center frequencies of bins.
    W = freqs[1:] - freqs[:-1]   # the differences between the frequencies
    W_l = torch.cat((W[:1], W))  # the difference between each center freq and the freq to the left
    W_r = torch.cat((W, W[-1:])) # the difference between each center freq and the freq to the right

    angles = C * t
    angles_r = (C - W_r) * t
    angles_l = (C + W_l) * t
    t2 = t**2
    scale_factor = 0.5 * (W_r + W_l)

    re = torch.where(t == 0., scale_factor,
                     (angles.cos() - angles_r.cos()) / (W_r * t2) + (angles.cos() - angles_l.cos()) / (W_l * t2))
    im = torch.where(t == 0., 0.0,
                     (angles.sin() - angles_r.sin()) / (W_r * t2) + (angles.sin() - angles_l.sin()) / (W_l * t2))


    if not scale:
        re = re / scale_factor
        im = im / scale_factor

    return torch.stack((re, im), dim=-1).to(dtype)


class AngularFreqBasis(nn.Module):
    """
    Computes and caches the angular-frequency basis used in relative position scoring.

    num_freqs: the number of frequencies of the sin and cos functions
    low_freq_factor: this is approximately the amount by which the lowest frequency will be
        less than the highest frequency, the highest frequency being the Nyquist (pi).
        The frequencies are close to a geometric series at higher frequency but linear
        at low frequency.
    """
    def __init__(self, num_freqs: int, low_freq_factor: float = 0.001):
        super().__init__()
        log_freqs = torch.linspace(math.log(low_freq_factor), math.log(1 + low_freq_factor), num_freqs)
        freqs = math.pi * (log_freqs.exp() - low_freq_factor)  # range from 0 to pi.
        freqs[0] = 0.0  # in case of roundoff
        self.register_buffer('freqs', freqs, persistent=False)

        self._cached_basis: Optional[Tensor] = None
        self._cached_seq_len: int = -1
        self._cached_left_context_len: int = -1

    def forward(self, seq_len: int, left_context_len: int, device: torch.device) -> Tensor:
        """
        Returns basis of shape (2 * seq_len + left_context_len - 1, 2 * num_freqs).

        The result is cached; if the requested (seq_len, left_context_len) fits
        within the cached range, the cached tensor is sliced rather than
        recomputed.
        """
        S = self._cached_seq_len
        L = self._cached_left_context_len
        if (self._cached_basis is not None
            and seq_len <= S
            and seq_len + left_context_len <= S + L):
            start = S + L - seq_len - left_context_len
            end = start + 2 * seq_len + left_context_len - 1
            return self._cached_basis[start:end]

        t = torch.arange(-(seq_len + left_context_len - 1), seq_len, device=device)
        basis = compute_angular_freq_basis_triangular(self.freqs, t, scale=False)
        # basis: (2 * seq_len + left_context_len - 1, num_freqs, 2)
        basis = basis.permute(0, 2, 1)
        # permute it because of how we did the low-pass initialization of weight, we want
        # the cos and sin parts to each be continuous ranges, not interleaved.
        basis = basis.reshape(basis.shape[0], -1)
        # basis: (2 * seq_len + left_context_len - 1, 2 * num_freqs)

        self._cached_basis = basis
        self._cached_seq_len = seq_len
        self._cached_left_context_len = left_context_len
        return basis


class RelPosScores(nn.Module):
    def __init__(self,
                 num_heads: int,
              pos_head_dim: int,
                 num_freqs: int):
        """
        Implementation of relative position scores; where conventional relative position scores
        would use sinusoids, we treat each sinusoid frequency as the central frequency of a
        triangular "bucket" (like the buckets in mel bins) of frequencies.   What this amounts
        to is that instead of a sinusoid we get something a bit like a sinusoid times a
        sinc-squared function (the sinc-squared function is the fourier transform of a triangular
        function).  Actually it's not the sinc-squared funtion, it's a slightly more complicated
        function than that because the "triangles" have uneven shapes, due to the center frequencies
        of the triangles not being evenly spaced.

        Args:
           num_heads: the number of heads
           pos_head_dim: the dimension of the head; in a conventionally structured model this would
                be identical to the query-dim but we make the "position query" independent of
                the main query and with a smaller dimension.
              num_freqs: the number of frequencies of the sin and cos functions
        """
        super().__init__()
        self.weight = nn.Parameter(0.04 * torch.randn(num_heads, pos_head_dim, 2 * num_freqs))
        with torch.no_grad():
            # initialize the weight in a low-pass way.  I think this is not so critical
            # actually, it may not matter.
            for _ in range(10):
                self.weight[:] = (2 ** -0.5) * (self.weight + self.weight.roll(1, dims=2))

        self.angular_freq_basis = AngularFreqBasis(num_freqs)

    def forward(self, p: Tensor, left_context_len: int = 0) -> Tensor:
        """
        Compute and return unnormalized log scores for relative position.
        Args:
           p: these are the position-queries, of shape (batch_size, num_heads, seq_len, pos_head_dim)
              (they are obtained via projection, just like the queries).
            left_context_len: length of left context, must be 0 for non-streaming forward and > 0 for streaming forward.
        Returns:
           scores: (batch_size, num_heads, dest_seq_len, src_seq_len), where dest_seq_len relates to the
         query and src_seq_len to the key.
         In non-streaming forward, dest_seq_len and src_seq_len are numerically equal to seq_len;
         in streaming forward, dest_seq_len is seq_len and src_seq_len is seq_len + left_context_len.
        """
        (batch_size, num_heads, seq_len, pos_head_dim) = p.shape

        basis = self.angular_freq_basis(seq_len, left_context_len, p.device)
        # basis: (2 * seq_len + left_context_len - 1, 2 * num_freqs)

        x = torch.matmul(self.weight, basis.t())
        assert x.shape == (num_heads, pos_head_dim, 2 * seq_len + left_context_len - 1)

        # with seq_len2 = 2 * seq_len + left_context_len - 1,
        # (batch, head, seq_len, pos_head_dim) x (1, head, pos_head_dim, seq_len2) -> (batch, head, seq_len, seq_len2)
        pos_weights = torch.matmul(p, x)

        # the following .as_strided() expression converts the last axis of pos_weights from relative
        # to absolute position.  This is all copied from our old conformer/zapformer code.
        if torch.jit.is_tracing():
            seq_len2 = pos_weights.shape[-1]
            rows = torch.arange(start=seq_len - 1, end=-1, step=-1)
            cols = torch.arange(left_context_len + seq_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_weights = pos_weights.reshape(-1, seq_len2)
            pos_weights = torch.gather(pos_weights, dim=1, index=indexes)
            pos_weights = pos_weights.reshape(batch_size, num_heads, seq_len, left_context_len + seq_len)
        else:
            pos_weights = pos_weights.as_strided(
                (batch_size, num_heads, seq_len, left_context_len + seq_len),
                (
                    pos_weights.stride(0),
                    pos_weights.stride(1),
                    pos_weights.stride(2) - pos_weights.stride(3),
                    pos_weights.stride(3),
                ),
                storage_offset=pos_weights.stride(3) * (seq_len - 1),
            )
        return pos_weights


def round_up_to_power_of_two(x):
    x = x - 1
    x = x | x >> 1
    x = x | x >> 2
    x = x | x >> 4
    x = x | x >> 8
    x = x | x >> 16
    x = x + 1
    return x



# FftConv was formerly used as the depthwise_conv module in ConvolutionModule.
# CAUTION: this is not used right now, we use BasisConv plus WeightedMean in
# parallel for the depthwise convolution in the ConvModule.  Using FftConv is
# actually just as good in WER terms and is also more efficient, versus (BasisConv
# plus WeightedMean); FftConv itself, should be about twice faster because it
# operates on a twice-shorter length than BasisConv since BasisConv pads for
# exactness.   For the overall training the speed difference is about 10%.
# The  reason we use BasisConv is because it is properly invariant to
# how we pad different-length sequences into a batch, while FftConv cannot
# be made to give exactly the same results independent of the batch size, because
# it treats the signal as repeating in time which depends on the FFT size which
# depends on the longest sequence in the batch.  Unfortunately, we don't know
# exactly how the model is going to be used and we don't want it to become
# deal-breaker that batching very-different-length sequences together in inference
# time could significantly affect the model results.  For image tasks,
# FftConv may still be useful (after suitable adaptation), because
# you wouldn't normally try to inference different size images in a batch.
class FftConv(nn.Module):
    def __init__(self,
                 num_channels: int,
                 params_per_channel: int,
                 bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(0.1 * torch.randn(num_channels, params_per_channel))
        # one factor of 2 is for (sin, cos); the other is to double the num representable freqs
        self.weight_proj = nn.Linear(params_per_channel, 4 * params_per_channel)
        if bias:
            self.bias = nn.Parameter(0.01 * torch.randn(num_channels))


    def forward(self,
                x: Tensor) -> Tensor:
        (seq_len, batch_size, num_channels) = x.shape

        # select a power of two that's >= seq_len // 8 and round up seq_len
        # to a multiple of that power.  This means that rounded_seq_len
        # will be of the form (2**n) * k where k <= 8, so it won't contain
        # many factors other than two; this will make the FFT more efficient
        # without adding an excessive amount of padding.
        power_of_two = max(1, round_up_to_power_of_two(seq_len // 8))
        rounded_seq_len = power_of_two * ((seq_len + power_of_two - 1) // power_of_two)


        with torch.amp.autocast('cuda', enabled=False):
            # do it in float32 because non power of two seq_len is not supported in half precision.
            x = torch.fft.rfft(x.to(torch.float32), dim=0, n=rounded_seq_len)
            # x: (num_freqs, batch_size, num_channels)
            N = x.shape[0]   # num freqs
            weight = 4. * self.weight
            weight = self.weight_proj(weight).reshape(num_channels, 2, -1)  # (num_channels, 2, 2 * params_per_channel)
            # this scale of 10 times is because of interactions with commonly
            # used optimizers, it's to help this module learn faster than it
            # otherwise would.
            weight = torch.nn.functional.interpolate(weight, N, mode='linear', align_corners=True)
            weight = torch.view_as_complex(weight.permute(2, 0, 1).contiguous())
            # weight: (N, num_channels)
            weight = weight.unsqueeze(1)  # (N, 1, num_channels)
            x = x * weight
            x = torch.fft.irfft(x, n=rounded_seq_len, dim=0)

        x = x[:seq_len]

        try:
            x = x + self.bias
        except AttributeError:
            pass

        return x


# convolution where we convolve with a combination of basis functions, the basis functions
# being based on linear interpolation in Fourier space-- in effect, each pair of basis functions
# corresponds to the real and imaginary coefficients for one triangular bin in Fourier space;
# in the time domain the triangular bin becomes a sinc^2 function and the frequency offset
# is just a complex exponential of which the real and imaginary coefficients give us sines and
# cosines.
def get_basis_funcs(seq_len: int,
                    num_freqs: int,
                    **kwargs
):
    """
    seq_len: the sequence length to which the basis functions are truncated; this is expected to
       be even
    num_freqs: the number of frequencies; the number of basis functions will be 2 * num_freqs,
        and note that the first pair of basis functions are special, because they are the
        (zero-freq; nyquist-freq) ones.
    kwargs: can be used for device

    Returns:
       basis functions of shape: (2 * num_freqs, seq_len)
    """
    assert seq_len % 2 == 0
    t = torch.cat((torch.arange(seq_len // 2, **kwargs),
                   torch.arange(-seq_len // 2, 0, **kwargs)), dim=0) # e.g. tensor([ 0,  1,  2,  3, -4, -3, -2, -1])
    # the second half of the "t" values are interpreted as the "negative half" of the time range--
    # the time range representing t values from -seq_len // 2 to seq_len // 2 - 1.
    # The way we use this will be to convolve it with a signal of size seq_len // 2 that
    # has been padded with zeroes of length seq_len // 2, and we want the result to be as if we padded with the basis
    # functions from -infinity to infinity.


    scaled_t = t * math.pi / num_freqs

    # "freqs" are the t values multiplied by the basis frequencies
    t_freqs = scaled_t * torch.arange(num_freqs + 1, **kwargs).unsqueeze(-1)
    # t_freqs: (num_freqs + 1, seq_len)

    # it's a sinc-squared envelope, as the frequency domain envelope is a
    # triangular, not a rectangular, function.  the factor of 0.5 comes
    # from the math
    sinc_arg = 0.5 * scaled_t
    envelope = torch.where(sinc_arg != 0.0, sinc_arg.sin() / sinc_arg, torch.ones_like(sinc_arg)) ** 2


    cos, sin = t_freqs.cos() * envelope, t_freqs.sin() * envelope
    #plt.plot(envelope)

    # the factor of 0.5 is because the other freqs would get "counted twice" due
    # to having two symmetric versions, the freqs at zero and the nyquist only have
    # one copy.  This ensures that if we give a coeff of all ones on all the
    # cos terms, we get (a scaled version of) the delta function.
    sin[0] = 0.5 * cos[-1]
    cos[0] = 0.5 * cos[0]
    # the sin coefficient of freq 0 and nyquist gives us nothing, so we use the cos
    # at the nyquist in this position.
    cos = cos[:num_freqs]
    sin = sin[:num_freqs]
    #scale = num_freqs ** -0.5  # scale to make the funcs have a value around 1.
    #cos = cos * scale
    #sin = sin * scale

    basis = torch.cat((cos, sin), dim=0)
    # basis: (2 * num_freqs, seq_len)

    #for i in range(num_freqs + 1):
    #    plt.plot(cos[i])
    #    plt.plot(sin[i])
    #    plt.show()
    return basis


def fourier_conv(x: Tensor, y: Tensor):
    # fourier based convolution of x and y, returns
    # something with the same sequence length as the shorter of
    # the two.
    # x, y: (seq_len, [1 or batch_size], num_channels)
    T = max(x.shape[0], y.shape[0])
    T_out = min(x.shape[0], y.shape[0])

    with torch.amp.autocast('cuda', enabled=False):
        x = x.to(torch.float)
        y = y.to(torch.float)
        X = torch.fft.rfft(x, dim=0, n=T)
        Y = torch.fft.rfft(y, dim=0, n=T)
        return torch.fft.irfft(X * Y, dim=0, n=T)[:T_out]

# fourier-based convolution, mem-efficient wrapper for fourier_conv.
class FourierConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return fourier_conv(x, y)

    @staticmethod
    def backward(ctx, ans_grad):
        # we could probably do a bit better than this by doing it manually
        x, y = ctx.saved_tensors
        with torch.enable_grad():
            x = x.detach()
            y = y.detach()
            x.requires_grad = True
            y.requires_grad = True
            ans = fourier_conv(x, y)
            ans.backward(gradient=ans_grad)
        return x.grad, y.grad


class WeightedMean(nn.Module):
    # this is like the core part of squeeze-and-excite: it computes a mean over time,
    # and then multiplies it by a learnable channel-specific weight.
    # we add this to a more conventional convolution; we found this was helpful because
    # normal convolution cannot do averaging-over-time since it does not know the
    # sequence length.
    def __init__(self,
                 num_channels: int,
                 causal: bool = False):
        super().__init__()
        self.causal = causal
        self.weights = nn.Parameter(0.1 * torch.randn(num_channels))

    def forward(self,
                x: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                apply_mask: bool = True) -> Tensor:
        """
        Compute weighted mean.
          x: (time, batch, channel)
        src_key_padding_mask: (batch, time), True for masked positions

        Returned shape: (time, batch, channel) if causal else (batch, channel)
        """
        T = x.shape[0]
        if self.causal:
            num_frames = torch.arange(1, T + 1, device=x.device)
            x_cumsum = torch.cumsum(x, dim=0)
            return x_cumsum / num_frames[:, None, None] * self.weights


        # assume x already masked, if mask is in use.
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.logical_not().to(torch.float)
            num_frames = mask.sum(dim=1)
            num_frames = num_frames.unsqueeze(-1).to(torch.float)

            if apply_mask:
                x = x * mask.t().unsqueeze(-1)

            # num_frames: (batch_size, 1)
            return x.mean(dim=0) * (T / num_frames) * self.weights
        else:
            return x.mean(dim=0) * self.weights

    def streaming_forward(
        self,
        x: Tensor,
        cached_sum: Tensor,
        cached_num_frames: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Streaming forward for causal weighted mean.

        Args:
            x: (time, batch, channel), the current chunk
            cached_sum: (1, batch, channel), cumulative sum from previous chunks
            cached_num_frames: (batch,), number of frames seen so far

        Returns:
            - output: (time, batch, channel)
            - new_cached_sum: (1, batch, channel)
            - new_cached_num_frames: (batch,)
        """
        T = x.shape[0]
        # cumsum within this chunk, then add the historical sum
        x_cumsum = torch.cumsum(x, dim=0) + cached_sum  # (T, batch, channel)

        # num_frames for each position in this chunk: (T, batch)
        num_frames = cached_num_frames.unsqueeze(0) + torch.arange(
            1, T + 1, device=x.device
        ).unsqueeze(1)  # (T, batch)

        output = x_cumsum / num_frames.unsqueeze(-1) * self.weights

        new_cached_sum = x_cumsum[-1:, :, :]  # (1, batch, channel)
        new_cached_num_frames = cached_num_frames + T  # (batch,)

        return output, new_cached_sum, new_cached_num_frames


class BasisConv(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_freqs: int,
                 params_per_channel: int):
        super().__init__()
        self.weight_proj = nn.Linear(params_per_channel, 2 * num_freqs)

        self.weight = nn.Parameter(0.05 * torch.randn(num_channels,
                                                      params_per_channel))


    def forward(self,
                x: Tensor) -> Tensor:
        (seq_len, batch_size, num_channels) = x.shape


        # round seq_len to a multiple of "round" to help ensure the FFT dimension
        # has plenty of powers of two; this will tend to make it more efficient.
        round = min(16, round_up_to_power_of_two(seq_len))
        seq_len_rounded = round * ((seq_len + round - 1) // round)

        # to ensure the answer is the same regardless of the amount of padding, we
        # pad the sequence to at least twice its initial length for purposes of
        # the FFT-based convolution.  Because we will view the basis functions
        # as going from t=-seq_len_rounded to t=seq_len_rounded - 1, this will
        # ensure that we never see "wrap-around" effects.
        T = 2 * seq_len_rounded

        num_freqs = self.weight_proj.weight.shape[0] // 2
        basis_funcs = get_basis_funcs(T, num_freqs, device=x.device)
        # basis_funcs: (2 * num_freqs, T)

        scale = num_freqs ** -0.5

        weight = scale * self.weight_proj(self.weight)
        # weight: (num_channels, 2 * num_freqs)
        channel_funcs = torch.matmul(weight, basis_funcs)
        # channel_funcs: (num_channels, T)


        # channel_funcs: (num_channels, T)
        channel_funcs = channel_funcs.t().unsqueeze(1)
        # channel_funcs: (T, 1, num_channels)

        return FourierConv.apply(channel_funcs, x)




class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zapformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()


        if not causal:
            assert kernel_size % 2 == 1
            self.depthwise_conv = nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )

        else:
            self.depthwise_conv = nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=0,  # will pad manually, on one side.
                bias=False,
            )
            self.left_pad = kernel_size - 1

        with torch.no_grad():
            # make the non-central convolution weights much smaller.
            k = kernel_size // 2
            self.depthwise_conv.weight[..., :k] *= 0.1
            self.depthwise_conv.weight[..., -k:] *= 0.1

        # add average-of-all-frames to the "convolution."; it has extra power vs the convolution
        # because the num frames differs between utterances.
        self.weighted_mean = WeightedMean(bottleneck_dim,
                                          causal=causal)

        self.out_proj = ActivationAndLinear(
            bottleneck_dim,
            channels,
            activation="SwashR",
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
        aux_loss_scale: float = 0.0,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).
        """
        input_scale = 3.
        x = self.in_proj(x * input_scale)  # (time, batch, 3*bottleneck_dim)

        x, y = x.chunk(2, dim=2)
        y = self.sigmoid(y)
        x = self.activation1(x)  # identity.

        # x: (time, batch, channels)
        # Caution: this module is not completely
        # invariant to the number of frames each sequence is padded with, since
        # the FFT-based convolution treats the signal as repeating.
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.t().unsqueeze(-1).expand_as(x), 0.0)


        wm = self.weighted_mean(x,
                                src_key_padding_mask,
                                apply_mask=False)  # just applied it.
        x = x.permute(1, 2, 0)  # (batch, bottleneck_dim, time)
        if self.causal:
            # Not support exporting a model for simulated streaming decoding
            assert not torch.jit.is_scripting() and not torch.jit.is_tracing()
            x_shape = x.shape
            x = torch.nn.functional.pad(x, (self.left_pad, 0))
            x = self.depthwise_conv(x)
            assert x.shape == x_shape, (x.shape, x_shape)
        else:
            x = self.depthwise_conv(x)  # x: (time, batch, bottleneck_dim)
        x = x.permute(2, 0, 1)  # (time, batch, bottleneck_dim)
        x = x + wm  # Add in the weighted-mean to the convolution; this adds extra power
        # because the utterances differ in length.

        x = x * y
        x = self.out_proj(x)  # (time, batch, channels)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cached_conv: Tensor,
        cached_wm_sum: Tensor,
        cached_wm_num_frames: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute convolution module in streaming mode.

        Args:
            x: Input tensor (#time, batch, channels).
            cached_conv: cached left context for depthwise_conv, of shape
                (#batch, channels, left_pad)
            cached_wm_sum: (1, batch, channels), cumulative sum for weighted_mean
            cached_wm_num_frames: (batch,), number of frames seen so far
            src_key_padding_mask: the mask for the src keys per batch (optional):
                (batch, #time), contains True in masked positions.

        Returns:
            - Output tensor (#time, batch, channels).
            - Updated cached_conv (#batch, channels, left_pad)
            - Updated cached_wm_sum (1, batch, channels)
            - Updated cached_wm_num_frames (batch,)
        """
        input_scale = 3.
        x = self.in_proj(x * input_scale)  # (time, batch, 3*bottleneck_dim)

        x, y = x.chunk(2, dim=2)
        y = self.sigmoid(y)

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.t().unsqueeze(-1).expand_as(x), 0.0)

        wm, cached_wm_sum, cached_wm_num_frames = self.weighted_mean.streaming_forward(
            x, cached_wm_sum, cached_wm_num_frames
        )

        x = x.permute(1, 2, 0)  # (batch, bottleneck_dim, time)

        x_shape = x.shape
        assert cached_conv.shape[-1] == self.left_pad, (cached_conv.shape[-1], self.left_pad)
        x = torch.cat([cached_conv, x], dim=2)
        cached_conv = x[..., -self.left_pad:]

        x = self.depthwise_conv(x)
        assert x.shape == x_shape, (x.shape, x_shape)

        x = x.permute(2, 0, 1)  # (time, batch, bottleneck_dim)
        x = x + wm

        x = x * y
        x = self.out_proj(x)  # (time, batch, channels)

        return x, cached_conv, cached_wm_sum, cached_wm_num_frames


def _test_zapformer_main(causal: bool = False):
    seq_len = 20
    # Just make sure the forward pass runs.

    input_dim = 50

    c = Zapformer(
        input_dim=input_dim,
        encoder_dim=(64, 96),
        num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
    )

    batch_size = 6
    seq_len = 21
    # Just make sure the forward pass runs.
    f, lengths = c(
        torch.randn(seq_len, batch_size, input_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        aux_loss_scale=1.0,
    )
    f.sum().backward()
    c.eval()
    x_ = c(
        torch.randn(seq_len, batch_size, input_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        aux_loss_scale=1.0,
    )
    x_  # to remove flake8 warnings

    logging.info(f"Zapformer forward test passed, causal={causal}")


def _test_zapformer_streaming():
    input_dim = 50
    batch_size = 2
    chunk_size = 32
    num_chunks = 10
    tail_chunk_size = 8
    seq_len = chunk_size * num_chunks + tail_chunk_size
    left_context_frames = 128

    model = Zapformer(
        input_dim=input_dim,
        encoder_dim=(64, 96, 128, 96),
        num_heads=(4, 4, 4, 4),
        conv_params=(31, 31, 15, 31), # it may be better to make these even if not in causal mode.
        downsampling_factor=(1, 2, 4, 2),
        causal=True,
        chunk_size=(chunk_size,),
        left_context_frames=(left_context_frames,),
    )

    model.compute_projection_overlap(verbose=True)

    model.eval()

    x_full = torch.randn(seq_len, batch_size, input_dim)
    x_lens_full = torch.full((batch_size,), seq_len, dtype=torch.int64)

    with torch.no_grad():
        out_full, out_lens_full = model(x_full, x_lens_full)

        caches = model.get_init_caches(batch_size=batch_size)

        out_chunks = []
        out_offset = 0
        processed_lens = torch.full((batch_size,), 0, dtype=torch.int64)

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            x_chunk = x_full[start:end]
            x_lens = torch.full((batch_size,), chunk_size, dtype=torch.int64)

            src_key_padding_mask = make_pad_mask(x_lens)
            # processed_mask is used to mask out initial states
            processed_mask = torch.arange(left_context_frames).expand(batch_size, left_context_frames)
            # (batch, left_context_size)
            processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
            # Update processed lengths
            processed_lens = processed_lens + x_lens

            # (batch, left_context_size + chunk_size)
            src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

            out_chunk, out_lens, caches = model.streaming_forward(
                x=x_chunk,
                x_lens=x_lens,
                caches=caches,
                src_key_padding_mask=src_key_padding_mask,
            )
            out_chunks.append(out_chunk)

            out_chunk_len = out_chunk.shape[0]
            expected_out = out_full[out_offset : out_offset + out_chunk_len]
            diff_chunk = torch.max(torch.abs(expected_out - out_chunk))
            logging.info(f"Chunk {i+1} | Input: {x_chunk.shape} -> Output: {out_chunk.shape} | Max diff: {diff_chunk}")
            assert torch.allclose(expected_out, out_chunk, atol=2e-5), f"Chunk {i+1} outputs do not match! Max diff: {diff_chunk}"

            out_offset += out_chunk_len

        x_tail = x_full[num_chunks * chunk_size:]
        x_lens_tail = torch.full((batch_size,), tail_chunk_size, dtype=torch.int64)
        src_key_padding_mask = make_pad_mask(x_lens_tail)
        # processed_mask is used to mask out initial states
        processed_mask = torch.arange(left_context_frames).expand(batch_size, left_context_frames)
        # (batch, left_context_size)
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        # Update processed lengths
        processed_lens = processed_lens + x_lens_tail

        # (batch, left_context_size + chunk_size)
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)
        out_tail, out_lens_tail, caches = model.streaming_forward(
            x=x_tail,
            x_lens=x_lens_tail,
            caches=caches,
            src_key_padding_mask=src_key_padding_mask,
        )
        out_chunks.append(out_tail)

        out_tail_len = out_tail.shape[0]
        expected_out_tail = out_full[out_offset : out_offset + out_tail_len]
        diff_tail = torch.max(torch.abs(expected_out_tail - out_tail))
        logging.info(f"Tail Chunk | Input: {x_tail.shape} -> Output: {out_tail.shape} | Max diff: {diff_tail}")
        assert torch.allclose(expected_out_tail, out_tail, atol=2e-5), f"Tail Chunk outputs do not match! Max diff: {diff_tail}"
        out_offset += out_tail_len

        out_stream_cat = torch.cat(out_chunks, dim=0)

        diff = torch.max(torch.abs(out_full - out_stream_cat))
        logging.info(f"Max abs diff between full forward and streaming forward: {diff}")

        assert torch.allclose(out_full, out_stream_cat, atol=2e-5), f"Outputs do not match! Max diff: {diff}"

    logging.info("Passed")



def _test_basis_conv():
    num_channels = 11
    f = BasisConv(num_channels=num_channels,
                  num_freqs=4,
                  params_per_channel=2)

    seq_len = 100
    subseq_len = 10 # will help visualize the effect
    batch_size = 2
    x = torch.cat((torch.randn(subseq_len, batch_size, num_channels),
                   torch.zeros(seq_len - subseq_len, batch_size, num_channels)),
                  dim=0)

    y = f(x)

    #plt.plot(x[:, 0, 0].detach())
    #plt.plot(y[:, 0, 0].detach())
    #plt.show()


    def rms(a):
        return (a**2).mean().item()
    print(f"rms(x)={rms(x)}, rms(y)={rms(y)}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # _test_basis_conv()
    # _test_zapformer_main(False)
    # _test_zapformer_main(True)
    _test_zapformer_streaming()
