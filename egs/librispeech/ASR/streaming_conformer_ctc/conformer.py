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


import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformer import Supervisions, Transformer, encoder_padding_mask


# from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/mask.py#L42
def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


class Conformer(Transformer):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): attention dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimention
        num_encoder_layers (int): number of encoder layers
        num_decoder_layers (int): number of decoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        normalize_before (bool): whether to use layer_norm before the first block.
        vgg_frontend (bool): whether to use vgg frontend.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        vgg_frontend: bool = False,
        use_feat_batchnorm: bool = False,
        causal: bool = False,
    ) -> None:
        super(Conformer, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            normalize_before=normalize_before,
            vgg_frontend=vgg_frontend,
            use_feat_batchnorm=use_feat_batchnorm,
        )

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = ConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            cnn_module_kernel,
            normalize_before,
            causal,
        )
        self.encoder = ConformerEncoder(encoder_layer, num_encoder_layers)
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(d_model)
        else:
            # Note: TorchScript detects that self.after_norm could be used inside forward()
            #       and throws an error without this change.
            self.after_norm = identity

    def run_encoder(
        self,
        x: Tensor,
        supervisions: Optional[Supervisions] = None,
        dynamic_chunk_training: bool = False,
        short_chunk_proportion: float = 0.5,
        chunk_size: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
          x:
            The model input. Its shape is (N, T, C).
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.
          dynamic_chunk_training:
            For training only.
            IF True, train with dynamic right context for some batches
            sampled with a distribution
            if False, train with full right context all the time.
          short_chunk_proportion:
            For training only.
            Proportion of samples that will be trained with dynamic chunk.
          chunk_size:
            For eval only.
            right context when evaluating test utts.
            -1 means all right context.
          simulate_streaming=False,
            For eval only.
            If true, the feature will be feeded into the model chunk by chunk.
            If false, the whole utts if feeded into the model together i.e. the
            model only foward once.


        Returns:
            Tensor: Predictor tensor of dimension (input_length, batch_size, d_model).
            Tensor: Mask tensor of dimension (batch_size, input_length)
        """
        if self.encoder.training:
            return self.train_run_encoder(
                x, supervisions, dynamic_chunk_training, short_chunk_proportion
            )
        else:
            return self.eval_run_encoder(
                x, supervisions, chunk_size, simulate_streaming
            )

    def train_run_encoder(
        self,
        x: Tensor,
        supervisions: Optional[Supervisions] = None,
        dynamic_chunk_training: bool = False,
        short_chunk_threshold: float = 0.5,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
          x:
            The model input. Its shape is (N, T, C).
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.
          dynamic_chunk_training:
            IF True, train with dynamic right context for some batches
            sampled with a distribution
            if False, train with full right context all the time.
          short_chunk_proportion:
            Proportion of samples that will be trained with dynamic chunk.
        """
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        src_key_padding_mask = encoder_padding_mask(x.size(0), supervisions)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(x.device)

        if dynamic_chunk_training:
            max_len = x.size(0)
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > (max_len * short_chunk_threshold):
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
            mask = ~subsequent_chunk_mask(
                size=x.size(0), chunk_size=chunk_size, device=x.device
            )
            x = self.encoder(
                x, pos_emb, mask=mask, src_key_padding_mask=src_key_padding_mask
            )  # (T, B, F)
        else:
            x = self.encoder(x, pos_emb, src_key_padding_mask=mask)  # (T, B, F)

        if self.normalize_before:
            x = self.after_norm(x)

        return x, src_key_padding_mask

    def eval_run_encoder(
        self,
        feature: Tensor,
        supervisions: Optional[Supervisions] = None,
        chunk_size: int = -1,
        simulate_streaming=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
          feature:
            The model input. Its shape is (N, T, C).
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.

        Returns:
            Tensor: Predictor tensor of dimension (input_length, batch_size, d_model).
            Tensor: Mask tensor of dimension (batch_size, input_length)
        """
        # feature.shape: N T C
        num_frames = feature.size(1)

        # As temporarily in icefall only subsampling_rate == 4 is supported,
        # following parameters are hard-coded here.
        # Change it accordingly if other subsamling_rate are supported.
        embed_left_context = 7
        embed_conv_right_context = 3
        subsampling_rate = 4
        stride = chunk_size * subsampling_rate
        decoding_window = embed_conv_right_context + stride

        # This is also only compatible to sumsampling_rate == 4
        length_after_subsampling = ((feature.size(1) - 1) // 2 - 1) // 2
        src_key_padding_mask = encoder_padding_mask(
            length_after_subsampling, supervisions
        )
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(feature.device)

        if chunk_size < 0:
            # non-streaming decoding
            x = self.encoder_embed(feature)
            x, pos_emb = self.encoder_pos(x)
            x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
            x = self.encoder(
                x, pos_emb, src_key_padding_mask=src_key_padding_mask
            )  # (T, B, F)
        else:
            if simulate_streaming:
                # simulate chunk_by_chunk streaming decoding
                # Results of this branch should be identical to following
                # "else" branch.
                # But this branch is a little slower
                # as the feature is feeded chunk by chunk

                # store the result  of chunk_by_chunk decoding
                encoder_output = []

                # caches
                pos_emb_positive = []
                pos_emb_negative = []
                pos_emb_central = None
                encoder_cache = [None for i in range(len(self.encoder.layers))]
                conv_cache = [None for i in range(len(self.encoder.layers))]

                # start chunk_by_chunk decoding
                offset = 0
                for cur in range(0, num_frames - embed_left_context + 1, stride):
                    end = min(cur + decoding_window, num_frames)
                    cur_feature = feature[:, cur:end, :]
                    cur_feature = self.encoder_embed(cur_feature)
                    cur_embed, cur_pos_emb = self.encoder_pos(cur_feature, offset)
                    cur_embed = cur_embed.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)

                    cur_T = cur_feature.size(1)
                    if cur == 0:
                        # for first chunk extract the central pos embedding
                        pos_emb_central = cur_pos_emb[0, (chunk_size - 1), :].view(
                            1, 1, -1
                        )
                        cur_T -= 1
                    pos_emb_positive.append(cur_pos_emb[0, :cur_T].flip(0))
                    pos_emb_negative.append(cur_pos_emb[0, -cur_T:])
                    assert pos_emb_positive[-1].size(0) == cur_T

                    pos_emb_pos = torch.cat(pos_emb_positive, dim=0).unsqueeze(0)
                    pos_emb_neg = torch.cat(pos_emb_negative, dim=0).unsqueeze(0)
                    cur_pos_emb = torch.cat(
                        [pos_emb_pos.flip(1), pos_emb_central, pos_emb_neg],
                        dim=1,
                    )

                    x = self.encoder.chunk_forward(
                        cur_embed,
                        cur_pos_emb,
                        src_key_padding_mask=src_key_padding_mask[
                            :, : offset + cur_embed.size(0)
                        ],
                        encoder_cache=encoder_cache,
                        conv_cache=conv_cache,
                        offset=offset,
                    )  # (T, B, F)
                    encoder_output.append(x)
                    offset += cur_embed.size(0)

                x = torch.cat(encoder_output, dim=0)
            else:
                # NOT simulate chunk_by_chunk decoding
                # Results of this branch should be identical to previous
                # simulate chunk_by_chunk decoding branch.
                # But this branch is faster.
                x = self.encoder_embed(feature)
                x, pos_emb = self.encoder_pos(x)
                x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
                mask = ~subsequent_chunk_mask(
                    size=x.size(0), chunk_size=chunk_size, device=x.device
                )
                x = self.encoder(
                    x,
                    pos_emb,
                    mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )  # (T, B, F)

        if self.normalize_before:
            x = self.after_norm(x)

        return x, src_key_padding_mask


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
        normalize_before: whether to use layer_norm before the first block.

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
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        causal: bool = False,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel, causal=causal)

        self.norm_ff_macaron = nn.LayerNorm(d_model)  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(d_model)  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

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

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """

        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(src))
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)

        # multi-headed self-attention module
        residual = src
        if self.normalize_before:
            src = self.norm_mha(src)
        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            src = self.norm_mha(src)

        # convolution module
        residual = src
        if self.normalize_before:
            src = self.norm_conv(src)
        src = residual + self.dropout(self.conv_module(src))
        if not self.normalize_before:
            src = self.norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)

        return src

    def chunk_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        encoder_cache: Optional[Tensor] = None,
        conv_cache: Optional[Tensor] = None,
        offset=0,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """

        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(src))
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)

        # multi-headed self-attention module
        residual = src
        if self.normalize_before:
            src = self.norm_mha(src)
        if encoder_cache is None:
            # src: [chunk_size, N, F] e.g. [8, 41, 512]
            key = src
            val = key
            encoder_cache = key
        else:
            key = torch.cat([encoder_cache, src], dim=0)
            val = key
            encoder_cache = key
        src_att = self.self_attn(
            src,
            key,
            val,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            offset=offset,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            src = self.norm_mha(src)

        # convolution module
        residual = src  # [chunk_size, N, F] e.g. [8, 41, 512]
        if self.normalize_before:
            src = self.norm_conv(src)
        if conv_cache is not None:
            src = torch.cat([conv_cache, src], dim=0)
        conv_cache = src

        src = self.conv_module(src)
        src = src[-residual.size(0) :, :, :]  # noqa: E203

        src = residual + self.dropout(src)
        if not self.normalize_before:
            src = self.norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)

        return src, encoder_cache, conv_cache


class ConformerEncoder(nn.TransformerEncoder):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None
    ) -> None:
        super(ConformerEncoder, self).__init__(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for mod in self.layers:
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def chunk_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        encoder_cache=None,
        conv_cache=None,
        offset=0,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for layer_index, mod in enumerate(self.layers):
            output, e_cache, c_cache = mod.chunk_forward(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                encoder_cache=encoder_cache[layer_index],
                conv_cache=conv_cache[layer_index],
                offset=offset,
            )
            encoder_cache[layer_index] = e_cache
            conv_cache[layer_index] = c_cache

        if self.norm is not None:
            output = self.norm(output)

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

    def extend_pe(self, x: Tensor, offset: int = 0) -> None:
        """Reset the positional encodings."""
        x_size_1 = offset + x.size(1)
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size_1 * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size_1, self.d_model)
        pe_negative = torch.zeros(x_size_1, self.d_model)
        position = torch.arange(0, x_size_1, dtype=torch.float32).unsqueeze(1)
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

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x, offset)
        x = x * self.xscale
        x_size_1 = offset + x.size(1)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x_size_1
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x_size_1,
        ]
        x_T = x.size(1)
        if offset > 0:
            pos_emb = torch.cat([pos_emb[:, :x_T], pos_emb[:, -x_T:]], dim=1)
        else:
            pos_emb = torch.cat(
                [
                    pos_emb[:, : (x_T - 1)],
                    self.pe[0, self.pe.size(1) // 2].view(1, 1, self.pe.size(-1)),
                    pos_emb[:, -(x_T - 1) :],  # noqa: E203
                ],
                dim=1,
            )

        return self.dropout(x), self.dropout(pos_emb)


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

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        offset=0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
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
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
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

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
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
            offset=offset,
        )

    def rel_shift(self, x: Tensor, offset=0) -> Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 == time1 + offset, since it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape
        time2 = time1 + offset
        assert n == 2 * time2 - 1
        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)

        return x.as_strided(
            (batch_size, num_heads, time1, time2),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        embed_dim_to_check: int,
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
        offset=0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim_to_check: total dimension of the model.
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

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(
                3, dim=-1
            )

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

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
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
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

        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim)
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
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)

        # (batch, 2*time1, head, d_k) --> (batch, head, d_k, 2*time -1)
        p = p.permute(0, 2, 3, 1)

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
        matrix_ac = torch.matmul(q_with_bias_u, k)  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p)  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd, offset=offset)  # [B, head, time1, time2]
        attn_output_weights = (
            matrix_ac + matrix_bd
        ) * scaling  # (batch, head, time1, time2)

        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, -1)

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
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


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
        causal: bool = False,
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
        # from https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/convolution.py#L41
        if causal:
            self.lorder = kernel_size - 1
            padding = 0  # manualy padding self.lorder zeros to the left later
        else:
            assert (kernel_size - 1) % 2 == 0
            self.lorder = 0
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
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x: Tensor) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        if self.lorder > 0:
            # manualy padding self.lorder zeros to the left
            # make depthwise_conv causal
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def identity(x):
    return x
