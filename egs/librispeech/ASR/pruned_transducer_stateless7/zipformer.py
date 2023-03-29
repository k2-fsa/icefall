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

import copy
import math
import warnings
import itertools
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from encoder_interface import EncoderInterface
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    Dropout3,
    SwooshL,
    SwooshR,
    ChunkCausalDepthwiseConv1d,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
    ScaleGrad,
)
from torch import Tensor, nn

from icefall.utils import make_pad_mask
from icefall.dist import get_rank


class Zipformer2(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.


        num_features (int): Number of input features, e.g. 40.
        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        encoder_unmasked_dim (int or Tuple[int]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        attention_share_layers: (int or Tuple[int]): how many successive layers share
           the same attention weights.   Must be at least 1.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
    """
    def __init__(
            self,
            num_features: int,
            output_downsampling_factor: int = 2,
            downsampling_factor: Tuple[int] = (2, 4),
            encoder_dim: Union[int, Tuple[int]] = 384,
            num_encoder_layers: Union[int, Tuple[int]] = 4,
            encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
            query_head_dim: Union[int, Tuple[int]]  = 24,
            pos_head_dim: Union[int, Tuple[int]]  = 4,
            value_head_dim: Union[int, Tuple[int]] = 12,
            num_heads: Union[int, Tuple[int]] = 8,
            attention_share_layers: Union[int, Tuple[int]] = 2,
            feedforward_dim: Union[int, Tuple[int]] = 1536,
            cnn_module_kernel: Union[int, Tuple[int]] = 31,
            pos_dim: int = 192,
            dropout: FloatLike = None,  # see code below for default
            warmup_batches: float = 4000.0,
            causal: bool = False,
            chunk_size: Tuple[int] = [-1],
            left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3),
                                     (20000.0, 0.1))

        # this is not the probability of skipping a layer.  It is the probability of
        # dropping out the "skip module" which allows the model to skip groups of
        # encoder stacks; when it's dropped out like this, it means we are forced
        # to take the "direct" (non-bypass) path.
        self.layer_skip_dropout_prob = ScheduledFloat((0.0, 0.5),
                                                      (warmup_batches, 0.025),
                                                      (20000.0, 0.0))

        def _to_tuple(x):
            """ Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.num_features = num_features  # int
        self.output_downsampling_factor = output_downsampling_factor # int
        self.downsampling_factor = downsampling_factor # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim) # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(encoder_unmasked_dim) # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        query_head_dim = _to_tuple(query_head_dim)
        value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        num_heads = _to_tuple(num_heads)
        attention_share_layers = _to_tuple(attention_share_layers)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u,d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T - 7) // 2, encoder_dims).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T - 7) // 2
        #   (2) embedding: num_features -> encoder_dims
        # In the normal configuration, we will downsample once more at the end
        # by a factor of 2, and most of the encoder stacks will run at a lower
        # sampling rate.
        self.encoder_embed = Conv2dSubsampling(num_features, encoder_dim[0],
                                               dropout=dropout)


        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):

            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
                attention_share_layers=attention_share_layers[i],
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )
                # we are adding a new attribute here.
                # this will be interpreted by get_named_parameter_groups_with_lrs().
                encoder.lr_scale = downsampling_factor[i] ** -0.33

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # initializes self.skip_layers and self.skip_modules
        self._init_skip_modules()

        self.downsample_output = SimpleDownsample(max(encoder_dim),
                                                  downsample=output_downsampling_factor,
                                                  dropout=dropout)


    def _init_skip_modules(self):
        """
        If self.downampling_factor = (1, 2, 4, 8, 4, 2), then at the input of layer
        indexed 4 (in zero indexing), with has subsapling_factor=4, we combine the output of
        layers 2 and 3; and at the input of layer indexed 5, which which has subsampling_factor=2,
        we combine the outputs of layers 1 and 5.
        """
        skip_layers = []
        skip_modules = []
        z = self.downsampling_factor
        for i in range(len(z)):
            if i <= 1 or z[i-1] <= z[i]:
                skip_layers.append(None)
                skip_modules.append(Identity())
            else:
                # TEMP
                for j in range(i-2, -1, -1):
                    if z[j] <= z[i] or j == 0:
                        # TEMP logging statement.
                        logging.info(f"At encoder stack {i}, which has downsampling_factor={z[i]}, we will "
                                     f"combine the outputs of layers {j} and {i-1}, with downsampling_factor={z[j]} and {z[i-1]}.")
                        skip_layers.append(j)
                        skip_modules.append(SimpleCombiner(self.encoder_dim[i-1],
                                                           min_weight=(0.0, 0.25)))
                        break
        self.skip_layers = skip_layers
        self.skip_modules = nn.ModuleList(skip_modules)

    def get_feature_masks(
            self,
            x: torch.Tensor) -> List[Union[float, Tensor]]:
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
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [ 1.0 ] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0

        max_downsampling_factor = max(self.downsampling_factor)

        num_frames_max = (num_frames0 + max_downsampling_factor - 1)

        # we divide the dropped-out feature dimensions into two equal groups;
        # the first group is dropped out with this probability, the second
        # group is dropped out with about twice this probability.
        feature_mask_dropout_prob = 0.125

        # frame_mask_max1 shape: (num_frames_max, batch_size, 1)
        frame_mask_max1 = (torch.rand(num_frames_max, batch_size, 1,
                                    device=x.device) >
                          feature_mask_dropout_prob).to(x.dtype)

        # frame_mask_max2 has additional frames masked, about twice the number.
        frame_mask_max2 = torch.logical_or(frame_mask_max1,
                                           (torch.rand(num_frames_max, batch_size, 1,
                                                       device=x.device) >
                                            feature_mask_dropout_prob).to(x.dtype))

        # dim: (num_frames_max, batch_size, 2)
        frame_mask_max = torch.cat((frame_mask_max1, frame_mask_max2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            ds = self.downsampling_factor[i]
            upsample_factor = (max_downsampling_factor // ds)

            frame_mask = (frame_mask_max.unsqueeze(1).expand(num_frames_max, upsample_factor,
                                                             batch_size, 2)
                          .reshape(num_frames_max * upsample_factor, batch_size, 2))
            num_frames = (num_frames0 + ds - 1) // ds
            frame_mask = frame_mask[:num_frames]
            feature_mask = torch.ones(num_frames, batch_size, self.encoder_dim[i],
                                      dtype=x.dtype, device=x.device)
            u1 = self.encoder_unmasked_dim[i]
            u2 = (u1 + self.encoder_dim[i]) // 2
            feature_mask[:, :, u1:u2] *= frame_mask[..., 0:1]
            feature_mask[:, :, u2:] *= frame_mask[..., 1:2]
            feature_masks.append(feature_mask)

        return feature_masks


    def get_chunk_info(self) -> Tuple[int, int]:
        """
        Returns chunk_size and left_context_chunks.
        """
        if not self.causal:
            return -1, -1
        chunk_size = random.choice(self.chunk_size)
        if chunk_size == -1:
            left_context_chunks = -1
        else:
            left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1
        return chunk_size, left_context_chunks


    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          chunk_size: Number of frames per chunk (only set this if causal == True).
              Must divide all elements of downsampling_factor.  At 50hz frame
              rate, i.e. after encoder_embed.  If not specified, no chunking.
          left_context_chunks: Number of left-context chunks for each chunk (affects
             attention mask); only set this if chunk_size specified.  If -1, there
             is no limit on the left context.  If not -1, require:
              left_context_chunks * context_size >= downsampling_factor[i] *
                                                    cnn_module_kernel[i] // 2.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")

        x = self.encoder_embed(x)

        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lengths = (x_lens - 7) // 2
        assert x.size(0) == lengths.max().item()
        src_key_padding_mask = make_pad_mask(lengths)

        outputs = []
        feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            if self.skip_layers[i] is not None:
                # this how we implement U-net-like skipping of some series of
                # stacks.  The layer_skip_dropout_prob is to discourage it from
                # completely ignoring the middle layers, especially early in
                # training,
                batch_size = x.shape[1]
                skip_x = self.skip_modules[i](outputs[self.skip_layers[i]], x)

                layer_skip_dropout_prob = float(self.layer_skip_dropout_prob)
                if self.training and layer_skip_dropout_prob > 0:
                    mask = (torch.rand((1, batch_size, 1), device=x.device) >
                            layer_skip_dropout_prob)
                    x = torch.where(mask, skip_x, x)
                else:
                    x = skip_x
            x = convert_num_channels(x, self.encoder_dim[i])
            x = module(x,
                       chunk_size=chunk_size,
                       feature_mask=feature_masks[i],
                       src_key_padding_mask=(None if src_key_padding_mask is None
                                             else src_key_padding_mask[...,::ds]),
                       attn_mask=attn_mask,
            )
            outputs.append(x)

        def get_full_dim_output():
            num_encoders = len(self.encoder_dim)
            assert len(outputs) == num_encoders
            output_dim = max(self.encoder_dim)
            output_pieces = [ outputs[-1] ]
            cur_dim = self.encoder_dim[-1]
            for i in range(num_encoders - 2, -1, -1):
                d = self.encoder_dim[i]
                if d > cur_dim:
                    this_output = outputs[i]
                    output_pieces.append(this_output[..., cur_dim:d])
                    cur_dim = d
            assert cur_dim == output_dim
            return torch.cat(output_pieces, dim=-1)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = get_full_dim_output()
        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2
        lengths = (lengths + 1) // 2

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths

    def _get_attn_mask(self, x: Tensor,
                       chunk_size: int,
                       left_context_chunks: int
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
            assert all (chunk_size * left_context_chunks >=
                        (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i]
                        for i in range(num_encoders))
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c,
                                     src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x),
                          (20000.0, ratio * x),
                          default=x)

def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))



class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """
    def __init__(
            self,
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            value_head_dim: int,
            feedforward_dim: int,
            dropout: FloatLike = 0.1,
            cnn_module_kernel: int = 31,
            causal: bool = False,
            # layer_skip_rate will be overwritten to change warmup begin and end times.
            # treating batch_index == 0.0 specially is just to get scan_pessimistic_batches_for_oom()
            # to work correctly.
            layer_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.05), default=0),
            attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
            ff2_skip_rate: FloatLike = 0.01,
            bypass_min: FloatLike = ScheduledFloat((0.0, 0.75), (20000.0, 0.2), default=0),
            bypass_max: FloatLike = 1.0,
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # probability of skipping the entire layer.
        self.layer_skip_rate = copy.deepcopy(layer_skip_rate)
        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)

        # min and max for self.bypass_scale, applied with probability 0.5 to avoid grads
        # ever becoming zero.
        self.bypass_min = copy.deepcopy(bypass_min)
        self.bypass_max = copy.deepcopy(bypass_max)
        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim=pos_dim, num_heads=num_heads,
            query_head_dim=query_head_dim, pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn = SelfAttention(embed_dim, num_heads,
                                        value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 3) // 4,
                                               dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 5) // 4,
                                               dropout)

        self.nonlin_attention = NonlinAttention(embed_dim,
                                                hidden_channels=3 * embed_dim // 4)

        self.conv_module = ConvolutionModule(embed_dim,
                                             cnn_module_kernel,
                                             causal=causal)


        #self.attention_squeeze = AttentionSqueeze(embed_dim, embed_dim // 2)

        self.norm = BiasNorm(embed_dim)

        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.balancer1 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.2, max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of AttentionSqueezeModule
        self.balancer_as = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(4.0, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)
        self.balancer2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.1, max_abs=4.0,
        )


    def remove_attention_weights(self):
        self.self_attn_weights = None

    def get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.bypass_min),
                                    max=float(self.bypass_max))
            layer_skip_rate = float(self.layer_skip_rate)
            if layer_skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > layer_skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            return ans

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask


    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask


    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        attn_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
         pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
         chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
       feature_mask: something that broadcasts with src, that we'll multiply `src`
              by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
         attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
               True means masked position. May be None.
    src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
             masked position.  May be None.

        Returns:
           (x, attn_weights) where x has the same shape as src, and attn_weights are of
              shape (num_heads, batch_size, seq_len, seq_len).
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        if self.self_attn_weights is not None:
            attn_weights = self.self_attn_weights(
                src,
                pos_emb=pos_emb,
                attn_mask=attn_mask,
                key_padding_mask=src_key_padding_mask,
            )
        # else rely on the ones passed in

        # use different heads for nonlin_attention and attention_squeeze, depending
        # whether this module has its on self_attn_weights submodule or is borrowing
        # attention weights from another one.
        head_offset = 0 if self.self_attn_weights is not None else 2

        self_attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        if True:
            selected_attn_weights = attn_weights[head_offset:head_offset+2]
            if random.random() < float(self.const_attention_rate):
                # Make attention weights constant.  The intention is to
                # encourage these modules to do something similar to an
                # averaging-over-time operation.
                # only need the mask, can just use the 1st one and expand later
                selected_attn_weights = selected_attn_weights[0:1]
                selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
                selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))
                selected_attn_weights = selected_attn_weights.expand(2, -1, -1, -1)


        na = self.balancer_na(self.nonlin_attention(src,
                                                    selected_attn_weights[0:1]))
        src = src + (na if self_attn_dropout_mask is None else na * self_attn_dropout_mask)

        src = src + self.feed_forward1(src)

        self_attn = self.self_attn(
            src, attn_weights)
        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        src = src + self.sequence_dropout(self.conv_module(src, chunk_size=chunk_size,
                                                           src_key_padding_mask=src_key_padding_mask),
                                          float(self.conv_skip_rate))

        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)),
                                          float(self.ff2_skip_rate))

        src = self.balancer1(src)
        src = self.norm(src)

        bypass_scale = self.get_bypass_scale(src.shape[1])
        # the next line equivalent to: src = src * bypass_scale + src_orig *
        # (1.0 - bypass_scale), but more memory efficient for backprop.
        src = src_orig + (src - src_orig) * bypass_scale

        src = self.balancer2(src)
        src = self.whiten(src)

        return src, attn_weights

class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """
    def __init__(
            self,
            encoder_layer: nn.Module,
            num_layers: int,
            pos_dim: int,
            dropout: float,
            warmup_begin: float,
            warmup_end: float,
            initial_layerdrop_rate: float = 0.5,
            final_layerdrop_rate: float = 0.05,
            attention_share_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15,
                                                        length_factor=3.0)

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end

        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            # treating batch_index=0.0 specially is just to get scan_pessimistic_batches_for_oom()
            self.layers[i].layer_skip_rate = ScheduledFloat((cur_begin, initial_layerdrop_rate),
                                                            (cur_end, final_layerdrop_rate),
                                                            default=0.0)
            cur_begin = cur_end
            if i % attention_share_layers != 0:
                self.layers[i].remove_attention_weights()

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)
        output = src

        rnd_seed = src.numel() + random.randint(0, 1000)

        output = output * feature_mask

        attn_weights = None

        for i, mod in enumerate(self.layers):
            output, attn_weights = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
                attn_weights=attn_weights,
            )

            output = output * feature_mask

        return output


class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """
    def __init__(self,
                 encoder: nn.Module,
                 dim: int,
                 downsample: int,
                 dropout: FloatLike):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim,
                                           downsample, dropout)
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = SimpleCombiner(dim, min_weight=(0.0, 0.25))


    def forward(self,
                src: Tensor,
                chunk_size: int = -1,
                feature_mask: Union[Tensor, float] = 1.0,
                attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds,::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[:src_orig.shape[0]]

        return self.out_combiner(src_orig, src)



class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """
    def __init__(self,
                 channels: int,
                 downsample: int,
                 dropout: FloatLike):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
            # right-pad src, repeating the last element.
            pad = d_seq_len * ds - seq_len
            src_extra = src[src.shape[0]-1:].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
            assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """
    def __init__(self,
                 num_channels: int,
                 upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src

class SimpleCombiner(torch.nn.Module):
    """
    A very simple way of combining 2 vectors of 2 different dims, via a
    learned weighted combination in the shared part of the dim.
    Args:
         dim1: the dimension of the first input, e.g. 256
         dim2: the dimension of the second input, e.g. 384.
    The output will have the same dimension as dim2.
    """
    def __init__(self,
                 dim: int,
                 min_weight: Tuple[float, float] = (0., 0.)):
        super(SimpleCombiner, self).__init__()
        initial_weight1 = 0.1
        self.weight1 = nn.Parameter(torch.full((dim,), initial_weight1))
        self.min_weight = min_weight

    def forward(self,
                src1: Tensor,
                src2: Tensor) -> Tensor:
        """
        src1: (*, other_dim)
        src2: (*, dim)

        Returns: a tensor of shape (*, dim)
        """
        assert src1.shape[:-1] == src2.shape[:-1]
        num_channels = src2.shape[-1]
        src1 = convert_num_channels(src1, num_channels)


        weight1 = limit_param_value(self.weight1,
                                    min=self.min_weight[0],
                                    max=1.0-self.min_weight[1],
                                    training=self.training)

        return src1 * weight1 + src2 * (1.0 - weight1)




class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works done by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """
    def __init__(
        self, embed_dim: int,
            dropout_rate: FloatLike,
            max_len: int = 1000,
            length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))



    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        T = x.size(0)
        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T-1), T,
                         device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = (self.embed_dim ** 0.5)
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = compression_length * x.sign() * ((x.abs() + compression_length).log() - math.log(compression_length))

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan() # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)


    def forward(self, x: torch.Tensor) -> Tensor:
        """Create positional encoding.

        Args:
            x (torch.Tensor): Input tensor (time, batch, `*`).

        Returns:
            positional embedding, of shape (1, 2*time-1, `*`).

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            self.pe.size(0) // 2
            - x.size(0)
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)



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
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            dropout: float = 0.0,
            pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                          (4000.0, 0.0))
    ) -> None:
        super().__init__()
        self.lr_scale = 0.9
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True,
                                    initial_scale=query_head_dim**-0.25)

        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=_whitening_schedule(3.0),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(key_head_dim * num_heads,
                                     channel_dim=-1,
                                     min_positive=0.4,
                                     max_positive=0.6,
                                     min_abs=0.0,
                                     max_abs=100.0,
                                     prob=0.025)


        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(pos_dim,
                                       num_heads * pos_head_dim,
                                       bias=False,
                                       initial_scale=0.05)


        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()


    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 2, pos_dim)
           chunk_size
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[...,0:query_dim]
        k = x[...,query_dim:2*query_dim]
        # p is the position-encoding query
        p = x[...,2*query_dim:]
        assert p.shape[-1] == num_heads * pos_head_dim


        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.


        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if not self.training or random.random() >= float(self.pos_emb_skip_rate):
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            pos_scores = pos_scores.as_strided((num_heads, batch_size, seq_len, seq_len),
                                               (pos_scores.stride(0),
                                                pos_scores.stride(1),
                                                pos_scores.stride(2)-pos_scores.stride(3),
                                                pos_scores.stride(3)),
                                               storage_offset=pos_scores.stride(3) * (seq_len - 1))

            attn_scores = attn_scores + pos_scores

        if self.training and random.random() < 0.1:
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
            attn_scores = penalize_abs_values_gt(attn_scores,
                                                 limit=25.0,
                                                 penalty=1.0e-04,
                                                 name=self.name)

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, seq_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights


    def _print_attn_entropy(
            self,
            attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1,2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


class SelfAttention(nn.Module):
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
        self.in_proj = nn.Linear(embed_dim,
                                 num_heads * value_head_dim,
                                 bias=True)

        self.out_proj = ScaledLinear(num_heads * value_head_dim,
                                     embed_dim, bias=True,
                                     initial_scale=0.05)

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)


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
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)     #  (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(
            seq_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x



class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model.
    """
    def __init__(self,
                 embed_dim: int,
                 feedforward_dim: int,
                 dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(feedforward_dim,
                                        channel_dim=-1,
                                        min_positive=0.3,
                                        max_positive=1.0,
                                        min_abs=0.75,
                                        max_abs=5.0)
        self.activation = SwooshL()
        # shared_dim=0 means we share the dropout mask along the time axis
        self.dropout = Dropout3(dropout, shared_dim=0)
        self.out_proj = ScaledLinear(feedforward_dim, embed_dim,
                                     initial_scale=0.1)

        self.out_whiten =  Whiten(num_groups=1,
                                  whitening_limit=_whitening_schedule(7.5),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.01)

    def forward(self,
                x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
            self,
            channels: int,
            hidden_channels: int,
    ) -> None:
        super().__init__()

        self.lr_scale = 0.95

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels,
                                     bias=True,
                                     initial_scale=0.05)



        self.whiten1 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)

        self.whiten2 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0, ratio=3.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)


    def forward(self,
                x: Tensor,
                attn_weights: Tensor,
    ) -> Tensor:
        """.
        Args:
           x: a Tensor of shape (seq_len, batch_size, num_channels)
attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
        Returns:
           a Tensor with the same shape as x
        """
        num_channels = x.shape[-1]
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)


        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """
    def __init__(
            self, channels: int, kernel_size: int, causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels, 2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.
        self.in_proj.lr_scale = 0.9

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.balancer1 = Balancer(
            bottleneck_dim, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity() # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity() # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = ChunkCausalDepthwiseConv1d(
            channels=bottleneck_dim,
            kernel_size=kernel_size) if causal else nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            groups=bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2)


        self.balancer2 = Balancer(
            bottleneck_dim, channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.activation3 = SwooshR()

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.out_proj = ScaledLinear(
            bottleneck_dim, channels,
            initial_scale=0.05,
        )


    def forward(self,
                x: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                chunk_size: int = -1,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=-1)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if chunk_size >= 0:
            assert self.causal, "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1) # (time, batch, channels)

        x = self.activation3(x)
        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)

        return x


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale



class ConvNeXt(nn.Module):
    """
    Our interpretation of the ConvNeXt module as used in https://arxiv.org/pdf/2206.14747.pdf
    """
    def __init__(self,
                 channels: int,
                 hidden_ratio: int = 3,
                 kernel_size: Tuple[int, int] = (7, 7),
                 layerdrop_rate: FloatLike = None):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        hidden_channels = channels * hidden_ratio
        if layerdrop_rate is None:
            layerdrop_rate = ScheduledFloat((0.0, 0.2), (20000.0, 0.015))
        self.layerdrop_rate = layerdrop_rate

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=padding)

        self.pointwise_conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1)

        self.hidden_balancer = Balancer(hidden_channels,
                                                  channel_dim=1,
                                                  min_positive=0.3,
                                                  max_positive=1.0,
                                                  min_abs=0.75,
                                                  max_abs=5.0)

        self.activation = SwooshL()
        self.pointwise_conv2 = ScaledConv2d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            initial_scale=0.01)

        self.out_balancer = Balancer(
            channels, channel_dim=1,
            min_positive=0.4, max_positive=0.6,
            min_abs=1.0, max_abs=6.0,
        )
        self.out_whiten = Whiten(num_groups=1,
                                 whitening_limit=5.0,
                                 prob=(0.025, 0.25),
                                 grad_scale=0.01)

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or not self.training:
            return self.forward_internal(x)
        layerdrop_rate = float(self.layerdrop_rate)

        if layerdrop_rate != 0.0:
            batch_size = x.shape[0]
            mask = torch.rand((batch_size, 1, 1, 1), dtype=x.dtype, device=x.device) > layerdrop_rate
        else:
            mask = None
        # turns out this caching idea does not work with --world-size > 1
        #return caching_eval(self.forward_internal, x, mask)
        return self.forward_internal(x, mask)


    def forward_internal(self,
                         x: Tensor,
                         layer_skip_mask: Optional[Tensor]  = None) -> Tensor:
        """
        x layout: (N, C, H, W), i.e. (batch_size, num_channels, num_frames, num_freqs)

        The returned value has the same shape as x.
        """
        bypass = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.hidden_balancer(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        if layer_skip_mask is not None:
            x = x * layer_skip_mask

        x = bypass + x
        x = self.out_balancer(x)
        x = x.transpose(1, 3)  # (N, W, H, C); need channel dim to be last
        x = self.out_whiten(x)
        x = x.transpose(1, 3)  # (N, C, H, W)

        return x



class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = (T-3)//2 - 2 == (T-7)//2

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 64,
        dropout: FloatLike = 0.1,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, (T-3)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
          bottleneck:
            bottleneck dimension for 1d squeeze-excite
        """
        assert in_channels >= 7
        super().__init__()

        # The ScaleGrad module is there to prevent the gradients
        # w.r.t. the weight or bias of the first Conv2d module in self.conv from
        # exceeding the range of fp16 when using automatic mixed precision (amp)
        # training.  (The second one is necessary to stop its bias from getting
        # a too-large gradient).

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
            ),
            ScaleGrad(0.2),
            Balancer(layer1_channels,
                               channel_dim=1,
                               max_abs=1.0),
            SwooshR(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            Balancer(layer2_channels,
                     channel_dim=1,
                     max_abs=4.0),
            SwooshR(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2), # (time, freq)
            ),
            Balancer(layer3_channels,
                     channel_dim=1,
                     max_abs=4.0),
            SwooshR(),
        )

        cur_width = (in_channels - 1) // 2

        # just one convnext layer
        self.convnext = ConvNeXt(layer3_channels, kernel_size=(7, 7))

        out_width = (((in_channels - 1) // 2) - 1) // 2

        self.out = nn.Linear(out_width * layer3_channels, out_channels)
        # use a larger than normal grad_scale on this whitening module; there is
        # only one such module, so there is not a concern about adding together
        # many copies of this extra gradient term.
        self.out_whiten = Whiten(num_groups=1,
                                 whitening_limit=_whitening_schedule(4.0),
                                 prob=(0.025, 0.25),
                                 grad_scale=0.02)

        # max_log_eps=0.0 is to prevent both eps and the output of self.out from
        # getting large, there is an unnecessary degree of freedom.
        self.out_norm = BiasNorm(out_channels)
        self.dropout = Dropout3(dropout, shared_dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        # scaling x by 0.1 allows us to use a larger grad-scale in fp16 "amp" (automatic mixed precision)
        # training, since the weights in the first convolution are otherwise the limiting factor for getting infinite
        # gradients.
        x = self.conv(x)
        x = self.convnext(x)

        # Now x is of shape (N, odim, ((T-3)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()

        x = x.transpose(1, 2).reshape(b, t, c * f)
        # now x: (N, ((T-1)//2 - 1))//2, out_width * layer3_channels))

        x = self.out(x)
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.out_whiten(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        return x



def _test_zipformer_main(causal: bool = False):
    feature_dim = 50
    batch_size = 5
    seq_len = 20
    feature_dim = 50
    # Just make sure the forward pass runs.

    c = Zipformer2(
        num_features=feature_dim, encoder_dim=(64,96), encoder_unmasked_dim=(48,64), num_heads=(4,4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,)
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
