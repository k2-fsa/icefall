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
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from encoder_interface import EncoderInterface
from s import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    LearnedScale,
)
from torch import Tensor, nn

from icefall.utils import make_pad_mask


class Conformer(EncoderInterface):
    """
    Args:
        num_features (int): Number of input features
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): embedding dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimention
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        vgg_frontend (bool): whether to use vgg frontend.
        warmup_batches (float): number of batches to warm up over
    """

    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        conformer_subsampling_factor: int = 4,
        d_model: Tuple[int] = (384, 384),
        encoder_unmasked_dim: int = 256,
        nhead: Tuple[int] = (8, 8),
        feedforward_dim: Tuple[int] = (1536, 2048),
        num_encoder_layers: Tuple[int] = (12, 12),
        dropout: float = 0.1,
        cnn_module_kernel: Tuple[int] = (31, 31),
        warmup_batches: float = 6000.0,
    ) -> None:
        super(Conformer, self).__init__()

        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        self.encoder_unmasked_dim = encoder_unmasked_dim
        assert 0 < d_model[0] <= d_model[1]
        self.d_model = d_model
        self.conformer_subsampling_factor = conformer_subsampling_factor

        assert encoder_unmasked_dim <= d_model[0] and encoder_unmasked_dim <= d_model[1]

        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model[0],
                                               dropout=dropout)

        encoder_layer1 = ConformerEncoderLayer(
            d_model[0],
            nhead[0],
            feedforward_dim[0],
            dropout,
            cnn_module_kernel[0],

        )
        # for the first third of the warmup period, we let the Conv2dSubsampling
        # layer learn something
        self.encoder1 = ConformerEncoder(
            encoder_layer1,
            num_encoder_layers[0],
            dropout,
            warmup_begin=0,
            warmup_end=warmup_batches / 2,
        )
        encoder_layer2 = ConformerEncoderLayer(
            d_model[1],
            nhead[1],
            feedforward_dim[1],
            dropout,
            cnn_module_kernel[1],

        )
        self.encoder2 = DownsampledConformerEncoder(
            ConformerEncoder(
                encoder_layer2,
                num_encoder_layers[1],
                dropout,
                warmup_begin=warmup_batches / 2,
                warmup_end=warmup_batches,
            ),
            input_dim=d_model[0],
            output_dim=d_model[1],
            downsample=conformer_subsampling_factor,
        )

        self.out_combiner = SimpleCombiner(d_model[0],
                                           d_model[1])

    def get_feature_mask(
            self,
            x: torch.Tensor) -> Tuple[Union[float, Tensor], Union[float, Tensor]]:
        """
        In eval mode, returns 1.0; in training mode, returns two randomized feature masks
        for the 1st and second encoders (which may run at different frame rates).
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.conformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (num_frames, batch_size, d_model0)
        """
        if not self.training:
            return 1.0, 1.0

        d_model0, d_model1 = self.d_model
        (num_frames0, batch_size, _d_model0) = x.shape
        assert d_model0 == _d_model0
        ds = self.conformer_subsampling_factor
        num_frames1 = ((num_frames0 + ds - 1) // ds)

        # on this proportion of the frames, drop out the extra features above
        # self.encoder_unmasked_dim.
        feature_mask_dropout_prob = 0.15

        # we only apply the random frame masking on 90% of sequences; we leave the remaining 10%
        # un-masked so that the model has seen un-masked data.
        sequence_mask_dropout_prob = 0.9

        # frame_mask is 0 with probability `feature_mask_dropout_prob`
        # frame_mask1 shape: (num_frames1, batch_size, 1)
        frame_mask1 = torch.logical_or(
            torch.rand(num_frames1, batch_size, 1, device=x.device) > feature_mask_dropout_prob,
            torch.rand(1, batch_size, 1, device=x.device) > sequence_mask_dropout_prob).to(x.dtype)

        feature_mask1 = torch.ones(num_frames1, batch_size, self.d_model[1],
                                   dtype=x.dtype, device=x.device)
        feature_mask1[:, :, self.encoder_unmasked_dim:] *= frame_mask1


        # frame_mask0 shape: (num_frames0, batch_size, 1)
        frame_mask0 = frame_mask1.unsqueeze(1).expand(num_frames1, ds, batch_size, 1).reshape(
            num_frames1 * ds, batch_size, 1)[:num_frames0]

        feature_mask0 = torch.ones(num_frames0, batch_size, self.d_model[0],
                                   dtype=x.dtype, device=x.device)
        feature_mask0[:, :, self.encoder_unmasked_dim:] *= frame_mask0

        return feature_mask0, feature_mask1


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
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, d_model)
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Caution: We assume the subsampling factor is 4!
            lengths = ((x_lens - 1) // 2 - 1) // 2
        assert x.size(0) == lengths.max().item()
        mask = make_pad_mask(lengths)

        feature_mask0, feature_mask1 = self.get_feature_mask(x)

        # x1:
        x1 = self.encoder1(
            x, feature_mask=feature_mask0, src_key_padding_mask=mask,
        )  # (T, N, C) where C == d_model[0]

        x2 = self.encoder2(
            x1, feature_mask=feature_mask1, src_key_padding_mask=mask,
        )  # (T, N, C) where C == d_model[1]

        x = self.out_combiner(x1, x2)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

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
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()

        self.d_model = d_model

        # we'll overwrite these warmup_begin and warmup_end values from init of
        # class ConformerEncoder.
        self.warmup_begin = 0.0
        self.warmup_end = 1000.0

        self.self_attn = RelPositionMultiheadAttention(
            d_model, nhead, dropout=dropout,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            ActivationBalancer(feedforward_dim,
                               channel_dim=-1, max_abs=10.0),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(feedforward_dim, d_model,
                         initial_scale=0.01),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            ActivationBalancer(feedforward_dim,
                               channel_dim=-1, max_abs=10.0),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(feedforward_dim, d_model,
                         initial_scale=0.01),
        )

        self.conv_module = ConvolutionModule(d_model,
                                             cnn_module_kernel)

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            d_model, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            max_abs=6.0,
            max_var_per_eig=0.2,
        )

    def get_warmup_value(self, warmup_count: float) -> float:
        """
        Returns a value that is 0 at the start of training and increases to 1.0 during
        a warmup period specified during model initialization.
        """
        if warmup_count < self.warmup_begin:
            return 0.0
        elif warmup_count > self.warmup_end:
            return 1.0
        else:
            return (warmup_count - self.warmup_begin) / (self.warmup_end - self.warmup_begin)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        attn_scores_in: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        warmup_count: float = 1.0e+10,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            attn_scores_in: something with the dimension fo attention weights (bsz, len, len, num_heads) that is
                   passed from layer to layer.
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        batch_split: if not None, this layer will only be applied to

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        src_orig = src

        # macaron style feed forward module
        src = src + self.feed_forward_macaron(src)

        # multi-headed self-attention module
        src_att, _, attn_scores_out = self.self_attn(
            src,
            pos_emb=pos_emb,
            attn_scores_in=attn_scores_in,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + src_att

        # convolution module
        src = src + self.conv_module(src, src_key_padding_mask=src_key_padding_mask)


        # feed forward module
        src = src + self.feed_forward(src)

        src = self.norm_final(self.balancer(src))

        warmup_value = self.get_warmup_value(warmup_count)
        if warmup_value < 1.0 and self.training:
            delta = src - src_orig
            keep_prob = 0.25 + 0.75 * warmup_value
            # the :1 means the mask is chosen per frame.
            delta = delta * (torch.rand_like(delta[...,:1]) < keep_prob)
            src = src_orig + delta


        return src, attn_scores_out


class ConformerEncoder(nn.Module):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = conformer_encoder(src)
    """
    def __init__(
            self,
            encoder_layer: nn.Module,
            num_layers: int,
            dropout: float,
            warmup_begin: float,
            warmup_end: float
    ) -> None:
        super().__init__()

        # keep track of how many times forward() has been called, for purposes of
        # warmup.  do this with a floating-point count because integer counts can
        # fail to survive model averaging.
        self.register_buffer('warmup_count', torch.tensor(0.0))

        # if this assert fails, increase the numbers in get_warmup_count().
        assert warmup_end <= 1000000.0

        self.encoder_pos = RelPositionalEncoding(encoder_layer.d_model,
                                                 dropout)

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end


        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_layers):
            self.layers[i].warmup_begin = cur_begin
            cur_begin += delta
            self.layers[i].warmup_end = cur_begin



    def get_warmup_count(self) -> float:
        """
        Returns a value that reflects how many times this function has been called in training mode.
        """
        ans = self.warmup_count.item()
        if self.training:
            if ans > 1000000.0:
                # this ensures that as the number of batches gets large, the warmup count cycles rather
                # than getting stuck at the smallest floating point value x such that x + 1 == x.
                # this is necessary because get_layers_to_drop() relies on the warmup count changing
                # on every batch.
                next_count = 500000.0
            else:
                next_count = ans + 1.0
            self.warmup_count.fill_(next_count)
        return ans


    def get_layers_to_drop(self, warmup_count: float):
        ans = set()
        if not self.training:
            return ans
        # We use a random number generator seeded from warmup_count because
        # if there are multiple training processes we want them to all drop the
        # same number of layers (not necessarily the same layers though).  This
        # will tend to minimize training time.
        rng = random.Random(int(warmup_count))
        num_layers = len(self.layers)

        # x is the expected number of layers to drop
        x = 0.075 * num_layers
        # integerize x in a way that preserves sxpectations.
        num_layers_to_drop = int(x) + int(rng.random() < (x - int(x)))
        while (len(ans) < num_layers_to_drop):
            # use random, not rng here, because we don't want the same specific layers to be dropped.
            ans.add(random.randrange(0, num_layers))
        return ans

    def forward(
        self,
        src: Tensor,
        feature_mask: Union[Tensor, float] = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: (x, x_no_combine), both of shape (S, N, E)
        """
        warmup_count = self.get_warmup_count()  # reflects number of training batches.
        pos_emb = self.encoder_pos(src)
        output = src

        outputs = []
        attn_scores = None

        layers_to_drop = self.get_layers_to_drop(warmup_count)

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
            if i in layers_to_drop:
                continue
            next_output, attn_scores = mod(
                output,
                pos_emb,
                attn_scores,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup_count=warmup_count,
            )
            # this seemed to be helpful...
            output = 0.5 * (next_output + output)

            output = output * feature_mask

        return output


class DownsampledConformerEncoder(nn.Module):
    r"""
    DownsampledConformerEncoder is a conformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output
    so that the output has the same shape as the input.
    """
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 output_dim: int,
                 downsample: int):
        super(DownsampledConformerEncoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = AttentionDownsample(input_dim, output_dim, downsample)
        self.encoder = encoder
        self.upsample = SimpleUpsample(output_dim, downsample)


    def forward(self,
                src: Tensor,
                feature_mask: Union[Tensor, float] = 1.0,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            mask: the mask for the src sequence (optional).  CAUTION: we need to downsample
                  this, if we are to support it.  Won't work correctly yet.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if mask is not None:
            mask = mask[::ds,::ds]
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[::ds]

        src = self.encoder(
            src, feature_mask=feature_mask, mask=mask, src_key_padding_mask=mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[:src_orig.shape[0]]

        return src


class AttentionDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: int):
        """
        Require out_channels > in_channels.
        """
        super(AttentionDownsample, self).__init__()
        self.query = nn.Parameter(torch.randn(in_channels) * (in_channels ** -0.5))

        # fill in the extra dimensions with a projection of the input
        if out_channels > in_channels:
            self.extra_proj = nn.Linear(in_channels * downsample,
                                        out_channels - in_channels,
                                        bias=False)
        else:
            self.extra_proj = None
        self.downsample = downsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds
        src_orig = src
        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
            # right-pad src, repeating the last element.
            pad = d_seq_len * ds - seq_len
            src_extra = src[src.shape[0]-1:].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
            assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        scores = (src * self.query).sum(dim=-1, keepdim=True)
        weights = scores.softmax(dim=1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)
        src = src.permute(0, 2, 1, 3).reshape(d_seq_len, batch_size, ds * in_channels)

        if self.extra_proj is not None:
            ans2 = self.extra_proj(src)
            ans = torch.cat((ans, ans2), dim=2)
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
        self.bias = nn.Parameter(torch.randn(upsample, num_channels) * 0.01)

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.bias.shape[0]
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src + self.bias.unsqueeze(1)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src

class SimpleCombiner(torch.nn.Module):
    """
    A very simple way of combining 2 vectors of 2 different dims, via a
    learned weighted combination in the shared part of the dim.
    Args:
         dim1: the dimension of the first input, e.g. 256
         dim2: the dimension of the second input, e.g. 384.  Require dim2 >= dim1.
    The output will have the same dimension as dim2.
    """
    def __init__(self,
                 dim1: int,
                 dim2: int):
        super(SimpleCombiner, self).__init__()
        assert dim2 >= dim1
        self.to_weight1 = nn.Parameter(torch.randn(dim1) * 0.01)
        self.to_weight2 = nn.Parameter(torch.randn(dim2) * 0.01)


    def forward(self,
                src1: Tensor,
                src2: Tensor) -> Tensor:
        """
        src1: (*, dim1)
        src2: (*, dim2)

        Returns: a tensor of shape (*, dim2)
        """
        assert src1.shape[:-1] == src2.shape[:-1]
        dim1 = src1.shape[-1]
        dim2 = src2.shape[-1]

        weight1 = (src1 * self.to_weight1).sum(dim=-1, keepdim=True)
        weight2 = (src2 * self.to_weight2).sum(dim=-1, keepdim=True)
        weight = (weight1 + weight2).sigmoid()

        src2_part1 = src2[...,:dim1]
        part1 = src1 * weight + src2_part1 * (1.0 - weight)
        part2 = src2[...,dim1:]
        return torch.cat((part1, part2), dim=-1)



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
        """Construct a PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(0), self.d_model)
        pe_negative = torch.zeros(x.size(0), self.d_model)
        position = torch.arange(0, x.size(0), dtype=torch.float32).unsqueeze(1)
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
            x (torch.Tensor): Input tensor (time, batch, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(0)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(0),
        ]
        return self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
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
        self.head_dim = embed_dim // (num_heads * 2)
        assert (
            self.head_dim * num_heads == self.embed_dim // 2
        ), "embed_dim//2 must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim // 2, bias=True)
        self.in_balancer = ActivationBalancer(3 * embed_dim // 2,
                                              channel_dim=-1, max_abs=5.0,
                                              max_var_per_eig=0.2)
        self.proj_balancer = ActivationBalancer(embed_dim // 2,
                                                channel_dim=-1, max_abs=10.0,
                                                min_positive=0.0, max_positive=1.0)
        self.out_proj = ScaledLinear(
            embed_dim // 2, embed_dim, bias=True, initial_scale=0.05
        )

        self.attn_scores_proj_in = nn.Parameter(torch.eye(num_heads))
        self.attn_scores_proj_out = nn.Parameter(torch.zeros(num_heads, num_heads))

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.pos_bias_u, -0.05, 0.05)
        nn.init.uniform_(self.pos_bias_v, -0.05, 0.05)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_scores_in: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""
        Args:
            x: input to be projected to query, key, value
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - x: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - attn_scores_in: :math:`(N, L, L, num_heads)`
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        x, weights, scores = self.multi_head_attention_forward(
            self.in_balancer(self.in_proj(x)),
            pos_emb,
            None if attn_scores_in is None else torch.matmul(attn_scores_in, self.attn_scores_proj_in),
            self.embed_dim,
            self.num_heads,
            self.in_proj.weight,
            self.in_proj.bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        if attn_scores_in is not None:
            attn_scores_out = torch.matmul(scores, self.attn_scores_proj_out)
            attn_scores_out = attn_scores_out + attn_scores_in
        else:
            # Here, add self.attn_scores_proj_in in order to make sure it has
            # a grad.
            attn_scores_out = torch.matmul(scores, self.attn_scores_proj_out +
                                           self.attn_scores_proj_in)
        return x, weights, attn_scores_out

    def rel_shift(self, x: Tensor) -> Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape
        assert n == 2 * time1 - 1
        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, time1, time1),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def multi_head_attention_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_scores_in: Optional[Tensor],
        embed_dim: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, _ = x.size()

        head_dim = embed_dim // (num_heads * 2)
        assert (
            head_dim * num_heads == embed_dim // 2
        ), "embed_dim must be divisible by num_heads"

        scaling = float(head_dim) ** -0.5


        # self-attention
        q, k, v = x.chunk(3, dim=-1)


        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, tgt_len, tgt_len]:
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct."
                    )
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    tgt_len,
                    tgt_len,
                ]:
                    raise RuntimeError(
                        "The size of the 3D attn_mask is not correct."
                    )
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()
                    )
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = (q * scaling).contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = self.proj_balancer(self.linear_pos(pos_emb)).view(pos_emb_bsz, -1, num_heads, head_dim)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + self.pos_bias_v).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(
            q_with_bias_u, k
        )  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p.transpose(-2, -1)
        )  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd)

        attn_output_weights = (
            matrix_ac + matrix_bd
        )  # (batch, head, time1, time2)

        attn_scores_out = attn_output_weights.permute(0, 2, 3, 1) # (batch, time1, time2, head)

        if attn_scores_in is not None:
            attn_output_weights = attn_output_weights + attn_scores_in.permute(0, 3, 1, 2)


        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, -1
        )

        assert list(attn_output_weights.size()) == [
            bsz * num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim // 2)
        )
        attn_output = nn.functional.linear(
            attn_output, out_proj_weight, out_proj_bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_scores_out
        else:
            return attn_output, None, attn_scores_out


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

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

        # after pointwise_conv1 we put x through a gated linear unit (nn.functional.glu).
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
        self.deriv_balancer1 = ActivationBalancer(
            2 * channels,
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
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

        self.deriv_balancer2 = ActivationBalancer(
            channels, channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.05,
        )

    def forward(self,
                x: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains bool in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
        """
        assert in_channels >= 7
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=1,
            ),
            ActivationBalancer(layer1_channels,
                               channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(layer2_channels,
                               channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(layer3_channels,
                               channel_dim=1),
            DoubleSwish(),
        )
        out_height = (((in_channels - 1) // 2 - 1) // 2)
        self.out = ScaledLinear(out_height * layer3_channels, out_channels)
        self.dropout = nn.Dropout(dropout)


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
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.dropout(x)
        return x

class AttentionCombine(nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.

    All but the last input will have a linear transform before we
    randomly combine them; these linear transforms will be initialized
    to the identity transform.

    The idea is that the list of Tensors will be a list of outputs of multiple
    conformer layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        num_channels: int,
        num_inputs: int,
        random_prob: float = 0.25,
        single_prob: float = 0.333,
    ) -> None:
        """
        Args:
          num_channels:
            the number of channels
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          random_prob:
            the probability with which we apply a nontrivial mask, in training
            mode.
         single_prob:
            the probability with which we mask to allow just a single
            module's output (in training)
        """
        super().__init__()

        self.random_prob = random_prob
        self.single_prob = single_prob
        self.weight  = torch.nn.Parameter(torch.zeros(num_channels,
                                                     num_inputs))
        self.bias = torch.nn.Parameter(torch.zeros(num_inputs))

        assert 0 <= random_prob <= 1, random_prob
        assert 0 <= single_prob <= 1, single_prob



    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels).  In test mode
          this is just the final input.
        """
        num_inputs = self.weight.shape[1]
        assert len(inputs) == num_inputs

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        weights = (stacked_inputs * self.weight).sum(dim=(1,)) + self.bias

        if random.random() < 0.002:
            logging.info(f"Average weights are {weights.softmax(dim=1).mean(dim=0)}")

        if self.training:
            # random masking..
            mask_start = torch.randint(low=1, high=int(num_inputs / self.random_prob),
                                       size=(num_frames,), device=weights.device).unsqueeze(1)
            # mask will have rows like: [ False, False, False, True, True, .. ]
            arange = torch.arange(num_inputs, device=weights.device).unsqueeze(0).expand(
                num_frames, num_inputs)
            mask = arange >= mask_start

            apply_single_prob = torch.logical_and(torch.rand(size=(num_frames, 1),
                                                             device=weights.device) < self.single_prob,
                                                  mask_start < num_inputs)
            single_prob_mask = torch.logical_and(apply_single_prob,
                                                 arange < mask_start - 1)

            mask = torch.logical_or(mask,
                                    single_prob_mask)

            weights = weights.masked_fill(mask, float('-inf'))
        weights = weights.softmax(dim=1)

        # (num_frames, num_channels, num_inputs) * (num_frames, num_inputs, 1) -> (num_frames, num_channels, 1),
        ans = torch.matmul(stacked_inputs, weights.unsqueeze(2))
        # ans: (*, num_channels)
        ans = ans.reshape(*tuple(inputs[0].shape[:-1]), num_channels)

        if __name__ == "__main__":
            # for testing only...
            print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans



def _test_random_combine():
    print("_test_random_combine()")
    num_inputs = 3
    num_channels = 50
    m = AttentionCombine(
        num_channels=num_channels,
        num_inputs=num_inputs,
        random_prob=0.5,
        single_prob=0.0)


    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_conformer_main():
    feature_dim = 50
    batch_size = 5
    seq_len = 20
    feature_dim = 50
    # Just make sure the forward pass runs.

    c = Conformer(
        num_features=feature_dim, d_model=(64,96), encoder_unmasked_dim=64, nhead=(4,4)
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings
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
    _test_random_combine()
    _test_conformer_main()
