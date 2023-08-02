#!/usr/bin/env python3
# Copyright (c)  2023 Xiaomi Corp.           (author: Wei Kang)
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
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ChunkCausalDepthwiseConv1d,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
)
from subformer import (
    BypassModule,
    CompactRelPositionalEncoding,
    LearnedDownsamplingModule,
    SubformerEncoder,
    SubformerEncoderLayer,
)
from zipformer import (
    DownsampledZipformer2Encoder,
    SimpleDownsample,
    SimpleUpsample,
    Zipformer2Encoder,
    Zipformer2EncoderLayer,
)
from torch import Tensor, nn


class Mixformer(EncoderInterface):
    def __init__(
        self,
        structure: str = "ZZS(S(S)S)SZ",
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (1, 1, 2, 2, 1),
        encoder_dim: Union[int, Tuple[int]] = (
            192, 
            192,
            256,
            384,
            512,
            384,
            256,
            192,
        ),
        num_encoder_layers: Union[int, Tuple[int]] = (
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ),
        encoder_unmasked_dim: Union[int, Tuple[int]] = (192, 192, 192),
        query_head_dim: Union[int, Tuple[int]] = (32,),
        value_head_dim: Union[int, Tuple[int]] = (12,),
        pos_head_dim: Union[int, Tuple[int]] = (4,),
        pos_dim: int = (48,),
        num_heads: Union[int, Tuple[int]] = (4,),
        feedforward_dim: Union[int, Tuple[int]] = (
            512,
            768,
            1024,
            1536,
            2048,
            1536,
            1024,
            768,
        ),
        cnn_module_kernel: Union[int, Tuple[int]] = (15, 31, 31),
        encoder_chunk_sizes: Tuple[Tuple[int, ...]] = ((128, 1024),),
        memory_dim: int = -1,
        dropout: Optional[FloatLike] = None,  # see code below for default
        warmup_batches: float = 4000.0,
        causal: bool = False,
    ) -> None:
        super(Mixformer, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        num_zip_encoders = len([s for s in structure if s == 'Z'])
        num_sub_encoders = len([s for s in structure if s == 'S'])
        num_encoders = num_zip_encoders + num_sub_encoders
        num_downsamplers = len([s for s in structure if s == '('])

        def _to_tuple(x, length):
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            assert isinstance(x, tuple)
            if len(x) == 1:
                x = x * length
            else:
                assert len(x) == length and isinstance(
                    x[0], int
                )
            return x

        self.output_downsampling_factor = output_downsampling_factor  # int
        self.downsampling_factor = (
            downsampling_factor
        ) = _to_tuple(downsampling_factor, num_zip_encoders + num_downsamplers)  # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim, num_encoders)  # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(
            encoder_unmasked_dim, num_zip_encoders
        )  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers, num_encoders)
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim, num_encoders)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim, num_encoders)
        pos_head_dim = _to_tuple(pos_head_dim, num_encoders)
        pos_dim = _to_tuple(pos_dim, num_encoders)
        self.num_heads = num_heads = _to_tuple(num_heads, num_encoders)
        feedforward_dim = _to_tuple(feedforward_dim, num_encoders)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(
            cnn_module_kernel, num_zip_encoders
        )
        encoder_chunk_sizes = _to_tuple(encoder_chunk_sizes, num_sub_encoders)

        self.causal = causal

        # for u, d in zip(encoder_unmasked_dim, encoder_dim):
            # assert u <= d

        # each one will be Zipformer2Encoder, DownsampledZipformer2Encoder,
        # SubformerEncoder or DownsampledSubformerEncoder
        zip_encoders = []
        sub_encoders = []
        downsamplers = []
        bypasses = []

        layer_indexes = []

        cur_max_dim = 0

        downsampling_factors_list = []
        def cur_downsampling_factor():
            c = 1
            for d in downsampling_factors_list: c *= d
            return c

        zip_encoder_dim = []
        zip_downsampling_factor = []
        for s in structure:
            if s == "Z":
                i = len(zip_encoders) + len(sub_encoders)
                j = len(zip_encoders)
                k = len(downsamplers) + len(zip_encoders)
                assert encoder_unmasked_dim[j] <= encoder_dim[i]
                zip_encoder_dim.append(encoder_dim[i])
                encoder_layer = Zipformer2EncoderLayer(
                    embed_dim=encoder_dim[i],
                    pos_dim=pos_dim[i],
                    num_heads=num_heads[i],
                    query_head_dim=query_head_dim[i],
                    pos_head_dim=pos_head_dim[i],
                    value_head_dim=value_head_dim[i],
                    feedforward_dim=feedforward_dim[i],
                    dropout=dropout,
                    cnn_module_kernel=cnn_module_kernel[j],
                    causal=causal,
                )

                # For the segment of the warmup period, we let the Conv2dSubsampling
                # layer learn something.  Then we start to warm up the other encoders.
                encoder = Zipformer2Encoder(
                    encoder_layer,
                    num_encoder_layers[i],
                    pos_dim=pos_dim[i],
                    dropout=dropout,
                    warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                    warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                    final_layerdrop_rate=0.035 * (downsampling_factor[k] ** 0.5),
                )

                if downsampling_factor[k] != 1:
                    encoder = DownsampledZipformer2Encoder(
                        encoder,
                        dim=encoder_dim[i],
                        downsample=downsampling_factor[k],
                        dropout=dropout,
                    )
                zip_downsampling_factor.append(downsampling_factor[k])
                layer_indexes.append(len(zip_encoders))
                zip_encoders.append(encoder)
            elif s == 'S':
                i = len(zip_encoders) + len(sub_encoders)
                j = len(sub_encoders)
                if len(sub_encoders) == 0:
                    cur_max_dim = encoder_dim[i]
                encoder_layer = SubformerEncoderLayer(
                    embed_dim=encoder_dim[i],
                    pos_dim=pos_head_dim[i],
                    num_heads=num_heads[i],
                    query_head_dim=query_head_dim[i],
                    value_head_dim=value_head_dim[i],
                    feedforward_dim=feedforward_dim[i],
                    memory_dim=memory_dim,
                    dropout=dropout,
                    causal=causal,
                )
                cur_max_dim = max(cur_max_dim, encoder_dim[i])
                encoder = SubformerEncoder(
                    encoder_layer,
                    num_encoder_layers[i],
                    embed_dim=cur_max_dim,
                    dropout=dropout,
                    chunk_sizes=encoder_chunk_sizes[j],
                    warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                    warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                    final_layerdrop_rate=0.035 * (cur_downsampling_factor() ** 0.5),
                )
                layer_indexes.append(len(sub_encoders))
                sub_encoders.append(encoder)
            elif s =='(':
                i = len(zip_encoders) + len(downsamplers)
                downsampler = LearnedDownsamplingModule(cur_max_dim,
                                                        downsampling_factor[i])
                downsampling_factors_list.append(downsampling_factor[i])
                layer_indexes.append(len(downsamplers))
                downsamplers.append(downsampler)
            else:
                assert s == ')'
                bypass = BypassModule(cur_max_dim, straight_through_rate=0.0)
                layer_indexes.append(len(bypasses))
                bypasses.append(bypass)
                downsampling_factors_list.pop()

        logging.info(f"cur_downsampling_factor={cur_downsampling_factor()}")

        self.zip_encoder_dim = zip_encoder_dim
        self.zip_downsampling_factor = zip_downsampling_factor
        self.layer_indexes = layer_indexes
        self.structure = structure
        self.zip_encoders = nn.ModuleList(zip_encoders)
        self.sub_encoders = nn.ModuleList(sub_encoders)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.bypasses = nn.ModuleList(bypasses)

        self.encoder_pos = CompactRelPositionalEncoding(64, pos_head_dim[0],
                                                        dropout_rate=0.15,
                                                        length_factor=1.0)

        self.downsample_output = SimpleDownsample(
            max(encoder_dim),
            downsample=output_downsampling_factor,
            dropout=dropout,
        )

    def _get_full_dim_output(self, outputs: List[Tensor]):
        num_encoders = len(self.zip_encoders) + 1
        assert len(outputs) == num_encoders
        output_dim = max(self.encoder_dim)
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[-1]
        for i in range(num_encoders - 2, -1, -1):
            d = list(outputs[i].shape)[-1]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim, (cur_dim, output_dim)
        return torch.cat(output_pieces, dim=-1)

    def get_feature_masks(self, x: Tensor) -> Union[List[float], List[Tensor]]:
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
             (1, batch_size, encoder_dims0)
        """
        num_encoders = len(self.zip_encoders)
        if not self.training:
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0

        feature_mask_dropout_prob = 0.125

        # mask1 shape: (1, batch_size, 1)
        mask1 = (
            torch.rand(1, batch_size, 1, device=x.device)
            > feature_mask_dropout_prob
        ).to(x.dtype)

        # mask2 has additional sequences masked, about twice the number.
        mask2 = torch.logical_and(
            mask1,
            (
                torch.rand(1, batch_size, 1, device=x.device)
                > feature_mask_dropout_prob
            ).to(x.dtype),
        )

        # dim: (1, batch_size, 2)
        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.zip_encoder_dim[i]
            feature_mask = torch.ones(
                1, batch_size, channels, dtype=x.dtype, device=x.device
            )
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

            feature_masks.append(feature_mask)

        return feature_masks

    def _get_attn_offset(self, x: Tensor, src_key_padding_mask: Optional[Tensor]) -> Optional[Tensor]:
        """
        Return attention offset of shape (1 or batch_size, seq_len, seq_len), interpreted as (1 or batch_size, tgt_seq_len,
            src_seq_len); this reflects masking, if causal == True, otherwise will be all zeros.

        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
         src_key_padding_mask: optional key-padding mask of shape (batch_size, seq_len) with True in masked positions.
        """
        seq_len, batch_size, _num_channels = x.shape

        ans = torch.zeros(batch_size, seq_len, seq_len, device=x.device)

        if self.causal:
            # t is frame index, shape (seq_len,)
            t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
            src_t = t
            tgt_t = t.unsqueeze(-1)
            attn_mask = (src_t > tgt_t)
            ans.masked_fill_(attn_mask, float('-inf'))

        if src_key_padding_mask is not None:
            ans.masked_fill_(src_key_padding_mask.unsqueeze(1), float('-inf'))
            # now ans: (batch_size, seq_len, seq_len).
        return ans


    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
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
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        outputs = []

        attn_offsets = [ self._get_attn_offset(x, src_key_padding_mask) ]
        pos_embs = [ self.encoder_pos(x) ]
        downsample_info = []

        if torch.jit.is_scripting():
            feature_masks = [1.0] * len(self.zip_encoders)
        else:
            feature_masks = self.get_feature_masks(x)

        for s, i in zip(self.structure, self.layer_indexes):
            if s == 'Z':
                encoder = self.zip_encoders[i]
                ds = self.zip_downsampling_factor[i]
                x = convert_num_channels(x, self.zip_encoder_dim[i])
                x = encoder(
                    x,
                    feature_mask=feature_masks[i],
                    src_key_padding_mask=(
                        None
                        if src_key_padding_mask is None
                        else src_key_padding_mask[..., ::ds]
                    ),
                )
                outputs.append(x)
            elif s == 'S':
                encoder = self.sub_encoders[i]  # one encoder stack
                x = encoder(x,
                            pos_embs[-1],
                            attn_offset=attn_offsets[-1])

                # only the last output of subformer will be used to combine the
                # final output.
                if i == len(self.sub_encoders) - 1:
                    outputs.append(x)
                # x will have the maximum dimension up till now, even if
                # `encoder` uses lower dim in its layers.
            elif s == '(':
                downsampler = self.downsamplers[i]

                indexes, weights, x_new = downsampler(x)
                downsample_info.append((indexes, weights, x))
                x = x_new

                pos_embs.append(downsampler.downsample_pos_emb(pos_embs[-1], indexes))

                attn_offsets.append(downsampler.downsample_attn_offset(attn_offsets[-1],
                                                                       indexes,
                                                                       weights))
            else:
                assert s == ')'  # upsample and bypass
                indexes, weights, x_orig = downsample_info.pop()
                _attn_offset = attn_offsets.pop()
                _pos_emb = pos_embs.pop()
                x_orig = convert_num_channels(x_orig, x.shape[-1])

                x = LearnedDownsamplingModule.upsample(x_orig, x, indexes, weights)

                bypass = self.bypasses[i]
                x = bypass(x_orig, x)

        # Only "balanced" structure is supported now
        assert len(downsample_info) == 0, len(downsample_info)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2
        if torch.jit.is_scripting():
            lengths = (x_lens + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lens + 1) // 2

        return x, lengths
