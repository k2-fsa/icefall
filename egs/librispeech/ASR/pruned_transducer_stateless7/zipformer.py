#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
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
import itertools
import logging
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
)
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    Identity,
    MaxEig,
    ScaledConv1d,
    Whiten,
    _diag,
    penalize_abs_values_gt,
    random_clamp,
    softmax,
)
from torch import Tensor, nn

from icefall.dist import get_rank
from icefall.utils import is_jit_tracing, make_pad_mask


class Zipformer(EncoderInterface):
    """
    Args:
        num_features (int): Number of input features
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
        num_features: int,
        output_downsampling_factor: int = 2,
        encoder_dims: Tuple[int] = (384, 384),
        attention_dim: Tuple[int] = (256, 256),
        encoder_unmasked_dims: Tuple[int] = (256, 256),
        zipformer_downsampling_factors: Tuple[int] = (2, 4),
        nhead: Tuple[int] = (8, 8),
        feedforward_dim: Tuple[int] = (1536, 2048),
        num_encoder_layers: Tuple[int] = (12, 12),
        dropout: float = 0.1,
        cnn_module_kernels: Tuple[int] = (31, 31),
        pos_dim: int = 4,
        warmup_batches: float = 4000.0,
    ) -> None:
        super(Zipformer, self).__init__()

        self.num_features = num_features
        assert 0 < encoder_dims[0] <= encoder_dims[1]
        self.encoder_dims = encoder_dims
        self.encoder_unmasked_dims = encoder_unmasked_dims
        self.zipformer_downsampling_factors = zipformer_downsampling_factors
        self.output_downsampling_factor = output_downsampling_factor

        # will be written to, see set_batch_count()
        self.batch_count = 0
        self.warmup_end = warmup_batches

        for u, d in zip(encoder_unmasked_dims, encoder_dims):
            assert u <= d, (u, d)

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T - 7)//2, encoder_dims).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T - 7)//2
        #   (2) embedding: num_features -> encoder_dims
        self.encoder_embed = Conv2dSubsampling(
            num_features, encoder_dims[0], dropout=dropout
        )

        # each one will be ZipformerEncoder or DownsampledZipformerEncoder
        encoders = []

        num_encoders = len(encoder_dims)
        for i in range(num_encoders):
            encoder_layer = ZipformerEncoderLayer(
                encoder_dims[i],
                attention_dim[i],
                nhead[i],
                feedforward_dim[i],
                dropout,
                cnn_module_kernels[i],
                pos_dim,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = ZipformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
            )

            if zipformer_downsampling_factors[i] != 1:
                encoder = DownsampledZipformerEncoder(
                    encoder,
                    input_dim=encoder_dims[i - 1] if i > 0 else encoder_dims[0],
                    output_dim=encoder_dims[i],
                    downsample=zipformer_downsampling_factors[i],
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # initializes self.skip_layers and self.skip_modules
        self._init_skip_modules()

        self.downsample_output = AttentionDownsample(
            encoder_dims[-1], encoder_dims[-1], downsample=output_downsampling_factor
        )

    def _get_layer_skip_dropout_prob(self):
        if not self.training:
            return 0.0
        batch_count = self.batch_count
        min_dropout_prob = 0.025

        if batch_count > self.warmup_end:
            return min_dropout_prob
        else:
            return 0.5 - (batch_count / self.warmup_end) * (0.5 - min_dropout_prob)

    def _init_skip_modules(self):
        """
        If self.zipformer_downampling_factors = (1, 2, 4, 8, 4, 2), then at the input of layer
        indexed 4 (in zero indexing), with has subsapling_factor=4, we combine the output of
        layers 2 and 3; and at the input of layer indexed 5, which which has subsampling_factor=2,
        we combine the outputs of layers 1 and 5.
        """
        skip_layers = []
        skip_modules = []
        z = self.zipformer_downsampling_factors
        for i in range(len(z)):
            if i <= 1 or z[i - 1] <= z[i]:
                skip_layers.append(None)
                skip_modules.append(SimpleCombinerIdentity())
            else:
                # TEMP
                for j in range(i - 2, -1, -1):
                    if z[j] <= z[i] or j == 0:
                        # TEMP logging statement.
                        logging.info(
                            f"At encoder stack {i}, which has downsampling_factor={z[i]}, we will "
                            f"combine the outputs of layers {j} and {i-1}, with downsampling_factors={z[j]} and {z[i-1]}."
                        )
                        skip_layers.append(j)
                        skip_modules.append(
                            SimpleCombiner(
                                self.encoder_dims[j],
                                self.encoder_dims[i - 1],
                                min_weight=(0.0, 0.25),
                            )
                        )
                        break
        self.skip_layers = skip_layers
        self.skip_modules = nn.ModuleList(skip_modules)

    def get_feature_masks(self, x: torch.Tensor) -> List[float]:
        # Note: The actual return type is Union[List[float], List[Tensor]],
        # but to make torch.jit.script() work, we use List[float]
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all encoder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoder dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_downsampling_factors times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (num_frames, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dims)
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dims[0] == _encoder_dims0, (
            self.encoder_dims,
            _encoder_dims0,
        )

        max_downsampling_factor = max(self.zipformer_downsampling_factors)

        num_frames_max = num_frames0 + max_downsampling_factor - 1

        feature_mask_dropout_prob = 0.15

        # frame_mask_max shape: (num_frames_max, batch_size, 1)
        frame_mask_max = (
            torch.rand(num_frames_max, batch_size, 1, device=x.device)
            > feature_mask_dropout_prob
        ).to(x.dtype)

        feature_masks = []
        for i in range(num_encoders):
            ds = self.zipformer_downsampling_factors[i]
            upsample_factor = max_downsampling_factor // ds

            frame_mask = (
                frame_mask_max.unsqueeze(1)
                .expand(num_frames_max, upsample_factor, batch_size, 1)
                .reshape(num_frames_max * upsample_factor, batch_size, 1)
            )
            num_frames = (num_frames0 + ds - 1) // ds
            frame_mask = frame_mask[:num_frames]
            feature_mask = torch.ones(
                num_frames,
                batch_size,
                self.encoder_dims[i],
                dtype=x.dtype,
                device=x.device,
            )
            u = self.encoder_unmasked_dims[i]
            feature_mask[:, :, u:] *= frame_mask
            feature_masks.append(feature_mask)

        return feature_masks

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
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
            - embeddings: its shape is (batch_size, output_seq_len, encoder_dims[-1])
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        lengths = (x_lens - 7) >> 1
        assert x.size(0) == lengths.max().item(), (x.shape, lengths, lengths.max())
        mask = make_pad_mask(lengths)

        outputs = []
        feature_masks = self.get_feature_masks(x)

        for i, (module, skip_module) in enumerate(
            zip(self.encoders, self.skip_modules)
        ):
            ds = self.zipformer_downsampling_factors[i]
            k = self.skip_layers[i]
            if isinstance(k, int):
                layer_skip_dropout_prob = self._get_layer_skip_dropout_prob()
                if torch.jit.is_scripting() or torch.jit.is_tracing():
                    x = skip_module(outputs[k], x)
                elif (not self.training) or random.random() > layer_skip_dropout_prob:
                    x = skip_module(outputs[k], x)
            x = module(
                x,
                feature_mask=feature_masks[i],
                src_key_padding_mask=None if mask is None else mask[..., ::ds],
            )
            outputs.append(x)

        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        lengths = (lengths + 1) >> 1

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths


class ZipformerEncoderLayer(nn.Module):
    """
    ZipformerEncoderLayer is made up of self-attn, feedforward and convolution networks.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        nhead: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        pos_dim: int = 4,
    ) -> None:
        super(ZipformerEncoderLayer, self).__init__()

        self.d_model = d_model

        # will be written to, see set_batch_count()
        self.batch_count = 0

        self.self_attn = RelPositionMultiheadAttention(
            d_model,
            attention_dim,
            nhead,
            pos_dim,
            dropout=0.0,
        )

        self.pooling = PoolingModule(d_model)

        self.feed_forward1 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.feed_forward2 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.conv_module1 = ConvolutionModule(d_model, cnn_module_kernel)

        self.conv_module2 = ConvolutionModule(d_model, cnn_module_kernel)

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
        # return dropout rate for the dynamic modules (self_attn, pooling, convolution); this
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
        src = src + self.feed_forward1(src)

        # dropout rate for submodules that interact with time.
        dynamic_dropout = self.get_dynamic_dropout_rate()

        # pooling module
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            src = src + self.pooling(src, key_padding_mask=src_key_padding_mask)
        elif random.random() >= dynamic_dropout:
            src = src + self.pooling(src, key_padding_mask=src_key_padding_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            src_att, attn_weights = self.self_attn(
                src,
                pos_emb=pos_emb,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + src_att

            src = src + self.conv_module1(
                src, src_key_padding_mask=src_key_padding_mask
            )

            src = src + self.feed_forward2(src)

            src = src + self.self_attn.forward2(src, attn_weights)

            src = src + self.conv_module2(
                src, src_key_padding_mask=src_key_padding_mask
            )
        else:
            use_self_attn = random.random() >= dynamic_dropout
            if use_self_attn:
                src_att, attn_weights = self.self_attn(
                    src,
                    pos_emb=pos_emb,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                )
                src = src + src_att

            if random.random() >= dynamic_dropout:
                src = src + self.conv_module1(
                    src, src_key_padding_mask=src_key_padding_mask
                )

            src = src + self.feed_forward2(src)
            if use_self_attn:
                src = src + self.self_attn.forward2(src, attn_weights)

            if random.random() >= dynamic_dropout:
                src = src + self.conv_module2(
                    src, src_key_padding_mask=src_key_padding_mask
                )

        src = src + self.feed_forward3(src)

        src = self.norm_final(self.balancer(src))

        delta = src - src_orig

        src = src_orig + delta * self.get_bypass_scale()

        return self.whiten(src)


class ZipformerEncoder(nn.Module):
    r"""ZipformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ZipformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
        >>> zipformer_encoder = ZipformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
    ) -> None:
        super().__init__()
        # will be written to, see set_batch_count() Note: in inference time this
        # may be zero but should be treated as large, we can check if
        # self.training is true.
        self.batch_count = 0
        self.warmup_begin = warmup_begin
        self.warmup_end = warmup_end
        # module_seed is for when we need a random number that is unique to the module but
        # shared across jobs.   It's used to randomly select how many layers to drop,
        # so that we can keep this consistent across worker tasks (for efficiency).
        self.module_seed = torch.randint(0, 1000, ()).item()

        self.encoder_pos = RelPositionalEncoding(encoder_layer.d_model, dropout)

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_layers):
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

    def forward(
        self,
        src: Tensor,
        # Note: The type of feature_mask should be Union[float, Tensor],
        # but to make torch.jit.script() work, we use `float` here
        feature_mask: float = 1.0,
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
        pos_emb = self.encoder_pos(src)
        output = src

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            layers_to_drop = []
        else:
            rnd_seed = src.numel() + random.randint(0, 1000)
            layers_to_drop = self.get_layers_to_drop(rnd_seed)

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                if i in layers_to_drop:
                    continue
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            output = output * feature_mask

        return output


class DownsampledZipformerEncoder(nn.Module):
    r"""
    DownsampledZipformerEncoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """

    def __init__(
        self, encoder: nn.Module, input_dim: int, output_dim: int, downsample: int
    ):
        super(DownsampledZipformerEncoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = AttentionDownsample(input_dim, output_dim, downsample)
        self.encoder = encoder
        self.upsample = SimpleUpsample(output_dim, downsample)
        self.out_combiner = SimpleCombiner(
            input_dim, output_dim, min_weight=(0.0, 0.25)
        )

    def forward(
        self,
        src: Tensor,
        # Note: the type of feature_mask should be Unino[float, Tensor],
        # but to make torch.jit.script() happ, we use float here
        feature_mask: float = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            mask: the mask for the src sequence (optional).  CAUTION: we need to downsample
                  this, if we are to support it.  Won't work correctly yet.
            src_key_padding_mask: the mask for the src keys per batch (optional).  Should
                  be downsampled already.

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
            mask = mask[::ds, ::ds]

        src = self.encoder(
            src,
            feature_mask=feature_mask,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)


class AttentionDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: int):
        """
        Require out_channels > in_channels.
        """
        super(AttentionDownsample, self).__init__()
        self.query = nn.Parameter(torch.randn(in_channels) * (in_channels**-0.5))

        # fill in the extra dimensions with a projection of the input
        if out_channels > in_channels:
            self.extra_proj = nn.Linear(
                in_channels * downsample, out_channels - in_channels, bias=False
            )
        else:
            self.extra_proj = None
        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample, could be 0 for onnx-export-compatibility
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds, (src.shape[0], d_seq_len, ds)

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        scores = (src * self.query).sum(dim=-1, keepdim=True)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            scores = penalize_abs_values_gt(scores, limit=10.0, penalty=1.0e-04)

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

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.bias = nn.Parameter(torch.randn(upsample, num_channels) * 0.01)

    def forward(self, src: Tensor) -> Tensor:
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


class SimpleCombinerIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        return src1


class SimpleCombiner(torch.nn.Module):
    """
    A very simple way of combining 2 vectors of 2 different dims, via a
    learned weighted combination in the shared part of the dim.
    Args:
         dim1: the dimension of the first input, e.g. 256
         dim2: the dimension of the second input, e.g. 384.
    The output will have the same dimension as dim2.
    """

    def __init__(self, dim1: int, dim2: int, min_weight: Tuple[float] = (0.0, 0.0)):
        super(SimpleCombiner, self).__init__()
        assert dim2 >= dim1, (dim2, dim1)
        self.weight1 = nn.Parameter(torch.zeros(()))
        self.min_weight = min_weight

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        """
        src1: (*, dim1)
        src2: (*, dim2)

        Returns: a tensor of shape (*, dim2)
        """
        assert src1.shape[:-1] == src2.shape[:-1], (src1.shape, src2.shape)

        weight1 = self.weight1
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if (
                self.training
                and random.random() < 0.25
                and self.min_weight != (0.0, 0.0)
            ):
                weight1 = weight1.clamp(
                    min=self.min_weight[0], max=1.0 - self.min_weight[1]
                )

        src1 = src1 * weight1
        src2 = src2 * (1.0 - weight1)

        src1_dim = src1.shape[-1]
        src2_dim = src2.shape[-1]
        if src1_dim != src2_dim:
            if src1_dim < src2_dim:
                src1 = torch.nn.functional.pad(src1, (0, src2_dim - src1_dim))
            else:
                src1 = src1[:src2_dim]

        return src1 + src2


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
        """Construct a PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        if is_jit_tracing():
            # 10k frames correspond to ~100k ms, e.g., 100 seconds, i.e.,
            # It assumes that the maximum input won't have more than
            # 10k frames.
            #
            # TODO(fangjun): Use torch.jit.script() for this module
            max_len = 10000
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
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

    def forward(self, x: torch.Tensor) -> Tensor:
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

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module, may be less or more than embed_dim
            but must be a multiple of num_heads.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        pos_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = attention_dim // num_heads
        self.pos_dim = pos_dim
        assert self.head_dim % 2 == 0, self.head_dim
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim,
            num_heads,
            attention_dim,
        )

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5, dividing it between the query and key.
        in_proj_dim = (
            2 * attention_dim  # query, key
            + attention_dim // 2  # value
            + pos_dim * num_heads  # positional encoding query
        )

        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=self.head_dim**-0.25
        )

        # self.whiten_values is applied on the values in forward();
        # it just copies the keys but prevents low-rank distribution by modifying grads.
        self.whiten_values = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )
        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            embed_dim, num_heads * pos_dim, bias=False, initial_scale=0.05
        )

        # the following are for diagnosics only, see --print-diagnostics option.
        # they only copy their inputs.
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

        self.out_proj = ScaledLinear(
            attention_dim // 2, embed_dim, bias=True, initial_scale=0.05
        )

        self.in_proj2 = nn.Linear(embed_dim, attention_dim // 2, bias=False)
        self.out_proj2 = ScaledLinear(
            attention_dim // 2, embed_dim, bias=True, initial_scale=0.05
        )
        # self.whiten_values2 is applied on the values in forward2()
        self.whiten_values2 = Whiten(
            num_groups=num_heads,
            whitening_limit=2.0,
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x: input to be projected to query, key, value
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - x: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
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

            - Returns: (attn_output, attn_weights)

             - attn_output: :math:`(S, N, E)` where S is the sequence length, N is the batch size,
                E is the embedding dimension.
              - attn_weights: :math:`(N * N, S, S)` where N is the batch size, H is the num-heads
                 and S is the sequence length.
        """
        x, weights = self.multi_head_attention_forward(
            self.in_proj(x),
            self.linear_pos(pos_emb),
            self.attention_dim,
            self.num_heads,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        return x, weights

    def multi_head_attention_forward(
        self,
        x_proj: Tensor,
        pos: Tensor,
        attention_dim: int,
        num_heads: int,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x_proj: the projected input, to be split into query, key, value.
            pos: head-specific biases arising from the positional embeddings.
            attention_dim: dimension inside attention mechanism
            num_heads: parallel attention heads.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - x: :math:`(L, N, 7 * A // 2)` where L is the target sequence length, N is the batch size, A is
              the attention dimension.  Will be split into (query, key, value, pos).
            - pos: :math:`(N, 2*L-1, A//2)` or :math:`(1, 2*L-1, A//2)` where L is the sequence
              length, N is the batch size, and A is the attention dim.
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
            - attn_weights: :math:`(N * H, S, S)` where N is the batch size,
              H is the num-heads, S is the sequence length.
        """

        seq_len, bsz, _ = x_proj.size()

        head_dim = attention_dim // num_heads
        pos_dim = self.pos_dim  # positional-encoding dim per head
        assert (
            head_dim * num_heads == attention_dim
        ), f"attention_dim must be divisible by num_heads: {head_dim}, {num_heads}, {attention_dim}"

        # self-attention
        q = x_proj[..., 0:attention_dim]
        k = x_proj[..., attention_dim : 2 * attention_dim]
        value_dim = attention_dim // 2
        v = x_proj[..., 2 * attention_dim : 2 * attention_dim + value_dim]
        # p is the position-encoding query, its dimension is num_heads*pos_dim..
        p = x_proj[..., 2 * attention_dim + value_dim :]

        k = self.whiten_keys(k)  # does nothing in the forward pass.
        v = self.whiten_values(v)  # does nothing in the forward pass.
        q = self.copy_query(q)  # for diagnostics only, does nothing.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

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
                if list(attn_mask.size()) != [1, seq_len, seq_len]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    seq_len,
                    seq_len,
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.reshape(seq_len, bsz, num_heads, head_dim)
        p = p.reshape(seq_len, bsz, num_heads, pos_dim)
        k = k.reshape(seq_len, bsz, num_heads, head_dim)
        v = v.reshape(seq_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == seq_len, "{} == {}".format(
                key_padding_mask.size(1), seq_len
            )

        q = q.permute(1, 2, 0, 3)  # (batch, head, time1, head_dim)
        p = p.permute(1, 2, 0, 3)  # (batch, head, time1, pos_dim)
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)

        seq_len2 = 2 * seq_len - 1
        pos = pos.reshape(1, seq_len2, num_heads, pos_dim).permute(0, 2, 3, 1)
        # pos shape now: (batch, head, pos_dim, seq_len2)

        # (batch, head, time1, pos_dim) x (1, head, pos_dim, seq_len2) -> (batch, head, time1, seq_len2)
        #  [where seq_len2 represents relative position.]
        pos_weights = torch.matmul(p, pos)
        # the following .as_strided() expression converts the last axis of pos_weights from relative
        # to absolute position.  I don't know whether I might have got the time-offsets backwards or
        # not, but let this code define which way round it is supposed to be.
        if torch.jit.is_tracing():
            (batch_size, num_heads, time1, n) = pos_weights.shape
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(seq_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_weights = pos_weights.reshape(-1, n)
            pos_weights = torch.gather(pos_weights, dim=1, index=indexes)
            pos_weights = pos_weights.reshape(batch_size, num_heads, time1, seq_len)
        else:
            pos_weights = pos_weights.as_strided(
                (bsz, num_heads, seq_len, seq_len),
                (
                    pos_weights.stride(0),
                    pos_weights.stride(1),
                    pos_weights.stride(2) - pos_weights.stride(3),
                    pos_weights.stride(3),
                ),
                storage_offset=pos_weights.stride(3) * (seq_len - 1),
            )

        # caution: they are really scores at this point.
        attn_output_weights = torch.matmul(q, k) + pos_weights

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if training and random.random() < 0.1:
                # This is a harder way of limiting the attention scores to not be too large.
                # It incurs a penalty if any of them has an absolute value greater than 50.0.
                # this should be outside the normal range of the attention scores.  We use
                # this mechanism instead of, say, a limit on entropy, because once the entropy
                # gets very small gradients through the softmax can become very small, and
                # some mechanisms like that become ineffective.
                attn_output_weights = penalize_abs_values_gt(
                    attn_output_weights, limit=25.0, penalty=1.0e-04
                )

        # attn_output_weights: (batch, head, time1, time2)
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, seq_len, seq_len
        )

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights = attn_output_weights.masked_fill(
                    attn_mask, float("-inf")
                )
            else:
                attn_output_weights = attn_output_weights + attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, seq_len, seq_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, seq_len, seq_len
            )

        # Using this version of softmax, defined in scaling.py,
        # should save a little of the memory used in backprop by, if
        # we are in automatic mixed precision mode (amp) == autocast,
        # only storing the half-precision output for backprop purposes.
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        # If we are using chunk-wise attention mask and setting a limited
        # num_left_chunks, the attention may only see the padding values which
        # will also be masked out by `key_padding_mask`. At this circumstances,
        # the whole column of `attn_output_weights` will be `-inf`
        # (i.e. be `nan` after softmax). So we fill `0.0` at the masking
        # positions to avoid invalid loss value below.
        if (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and key_padding_mask is not None
        ):
            if attn_mask.size(0) != 1:
                attn_mask = attn_mask.view(bsz, num_heads, seq_len, seq_len)
                combined_mask = attn_mask | key_padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                # attn_mask.shape == (1, tgt_len, src_len)
                combined_mask = attn_mask.unsqueeze(0) | key_padding_mask.unsqueeze(
                    1
                ).unsqueeze(2)

            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, seq_len, seq_len
            )
            attn_output_weights = attn_output_weights.masked_fill(combined_mask, 0.0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, seq_len, seq_len
            )

        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, seq_len, head_dim // 2]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, attention_dim // 2)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        return attn_output, attn_output_weights

    def forward2(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Second forward function, where we re-use the attn_weights returned by the first forward function
        but with different input.
        Args:
               x: input, of shape (seq_len, batch_size, embed_dim)
           attn_weights: attention weights returned by forward(), of shape (batch_size * num_heads, seq_len, seq_len)
        Returns:
               output of the same shape as x, i.e. (seq_len, batch_size, embed_dim)
        """
        num_heads = self.num_heads
        (seq_len, bsz, embed_dim) = x.shape
        head_dim = self.attention_dim // num_heads
        # v: (tgt_len, bsz, embed_dim // 2)
        v = self.in_proj2(x)
        v = self.whiten_values2(v)  # does nothing in the forward pass.
        v = v.reshape(seq_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        # now v: (bsz * num_heads, seq_len, head_dim // 2)
        attn_output = torch.bmm(attn_weights, v)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if random.random() < 0.001 or __name__ == "__main__":
                self._print_attn_stats(attn_weights, attn_output)

        # attn_output: (bsz * num_heads, seq_len, head_dim)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, self.attention_dim // 2)
        )
        # returned value is of shape (seq_len, bsz, embed_dim), like x.
        return self.out_proj2(attn_output)

    def _print_attn_stats(self, attn_weights: Tensor, attn_output: Tensor):
        # attn_weights: (batch_size * num_heads, seq_len, seq_len)
        # attn_output: (bsz * num_heads, seq_len, head_dim)
        (n, seq_len, head_dim) = attn_output.shape
        num_heads = self.num_heads
        bsz = n // num_heads

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_output = attn_output.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .reshape(bsz, num_heads, seq_len)
                    .mean(dim=(0, 2))
                )
                attn_output = attn_output.reshape(bsz, num_heads, seq_len, head_dim)
                attn_output = attn_output.permute(1, 0, 2, 3).reshape(
                    num_heads, bsz * seq_len, head_dim
                )
                attn_output_mean = attn_output.mean(dim=1, keepdim=True)
                attn_output = attn_output - attn_output_mean
                attn_covar = torch.matmul(attn_output.transpose(1, 2), attn_output) / (
                    bsz * seq_len
                )
                # attn_covar: (num_heads, head_dim, head_dim)
                # eigs, _ = torch.symeig(attn_covar)
                # logging.info(f"attn_weights_entropy = {attn_weights_entropy}, output_eigs = {eigs}")

                attn_covar = _diag(attn_covar).mean(dim=1)  # (num_heads,)
                embed_dim = self.in_proj2.weight.shape[1]
                in_proj_covar = (
                    self.in_proj2.weight.reshape(num_heads, head_dim, embed_dim) ** 2
                ).mean(dim=(1, 2))
                out_proj_covar = (
                    self.out_proj2.weight.reshape(embed_dim, num_heads, head_dim) ** 2
                ).mean(dim=(0, 2))
                logging.info(
                    f"attn_weights_entropy = {attn_weights_entropy}, covar={attn_covar}, in_proj_covar={in_proj_covar}, out_proj_covar={out_proj_covar}"
                )


class PoolingModule(nn.Module):
    """
    Averages the input over the time dimension and project with a square matrix.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = ScaledLinear(d_model, d_model, initial_scale=0.1, bias=False)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        """
        Args:
           x: a Tensor of shape (T, N, C)
          key_padding_mask: a Tensor of bool, of shape (N, T), with True in masked
            positions.
        Returns:
           a Tensor of shape (1, N, C)
        """
        if key_padding_mask is not None:
            if torch.jit.is_tracing():
                pooling_mask = (~key_padding_mask).to(x.dtype)
            else:
                pooling_mask = key_padding_mask.logical_not().to(x.dtype)  # (N, T)
            pooling_mask = pooling_mask / pooling_mask.sum(dim=1, keepdim=True)
            pooling_mask = pooling_mask.transpose(0, 1).contiguous().unsqueeze(-1)
            # now pooling_mask: (T, N, 1)
            x = (x * pooling_mask).sum(dim=0, keepdim=True)
        else:
            num_frames = x.shape[0]
            pooling_mask = 1.0 / num_frames
            x = (x * pooling_mask).sum(dim=0, keepdim=True)

        x = self.proj(x)
        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer model."""

    def __init__(self, d_model: int, feedforward_dim: int, dropout: float):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(d_model, feedforward_dim)
        self.balancer = ActivationBalancer(
            feedforward_dim, channel_dim=-1, max_abs=10.0, min_prob=0.25
        )
        self.activation = DoubleSwish()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = ScaledLinear(feedforward_dim, d_model, initial_scale=0.01)

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.balancer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0, kernel_size

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
            channel_dim=1,
            max_abs=10.0,
            min_positive=0.05,
            max_positive=1.0,
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
            channels,
            channel_dim=1,
            min_positive=0.05,
            max_positive=1.0,
            max_abs=20.0,
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

    def forward(
        self,
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
        layer3_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, (T-7)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer2_channels:
            Number of channels in layer2
          layer3_channels:
            Number of channels in layer3
        """
        assert in_channels >= 7, in_channels
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
            ),
            ActivationBalancer(layer1_channels, channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            ActivationBalancer(layer2_channels, channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2),  # (time, freq)
            ),
            ActivationBalancer(layer3_channels, channel_dim=1),
            DoubleSwish(),
        )
        out_height = (((in_channels - 1) // 2) - 1) // 2
        self.out = ScaledLinear(out_height * layer3_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, (T-7)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, (T-7)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        # Now x is of shape (N, (T-7)//2, odim)
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
    zipformer layers.  This has a similar effect as iterated loss. (See:
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
        self.weight = torch.nn.Parameter(torch.zeros(num_channels, num_inputs))
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

        scores = (stacked_inputs * self.weight).sum(dim=(1,)) + self.bias

        if random.random() < 0.002:
            logging.info(f"Average scores are {scores.softmax(dim=1).mean(dim=0)}")

        if self.training:
            # random masking..
            mask_start = torch.randint(
                low=1,
                high=int(num_inputs / self.random_prob),
                size=(num_frames,),
                device=scores.device,
            ).unsqueeze(1)
            # mask will have rows like: [ False, False, False, True, True, .. ]
            arange = (
                torch.arange(num_inputs, device=scores.device)
                .unsqueeze(0)
                .expand(num_frames, num_inputs)
            )
            mask = arange >= mask_start

            apply_single_prob = torch.logical_and(
                torch.rand(size=(num_frames, 1), device=scores.device)
                < self.single_prob,
                mask_start < num_inputs,
            )
            single_prob_mask = torch.logical_and(
                apply_single_prob, arange < mask_start - 1
            )

            mask = torch.logical_or(mask, single_prob_mask)

            scores = scores.masked_fill(mask, float("-inf"))

        if self.training and random.random() < 0.1:
            scores = penalize_abs_values_gt(scores, limit=10.0, penalty=1.0e-04)

        weights = scores.softmax(dim=1)

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
        single_prob=0.0,
    )

    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_zipformer_main():
    feature_dim = 50
    batch_size = 5
    seq_len = 20
    feature_dim = 50
    # Just make sure the forward pass runs.

    c = Zipformer(
        num_features=feature_dim,
        encoder_dims=(64, 96),
        encoder_unmasked_dims=(48, 64),
        nhead=(4, 4),
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    assert ((seq_len - 7) // 2 + 1) // 2 == f[0].shape[1], (seq_len, f.shape[1])
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


def _test_conv2d_subsampling():
    num_features = 80
    encoder_dims = 384
    dropout = 0.1
    encoder_embed = Conv2dSubsampling(num_features, encoder_dims, dropout=dropout)
    for i in range(20, 40):
        x = torch.rand(2, i, num_features)
        y = encoder_embed(x)
        assert (x.shape[1] - 7) // 2 == y.shape[1], (x.shape[1], y.shape[1])


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_random_combine()
    _test_zipformer_main()
    _test_conv2d_subsampling()
