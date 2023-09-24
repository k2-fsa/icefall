# Copyright    2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Copyright    2022  Xiaomi Corp.                              (author: Quandong Wang)
#              2023  Johns Hopkins University                  (author: Dongji Gao)
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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from attention import MultiheadAttention
from label_smoothing import LabelSmoothingLoss
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledEmbedding,
    ScaledLinear,
)
from subsampling import Conv2dSubsampling
from torch.nn.utils.rnn import pad_sequence

# Note: TorchScript requires Dict/List/etc. to be fully typed.
Supervisions = Dict[str, torch.Tensor]


class Transformer(nn.Module):
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
        layer_dropout: float = 0.075,
    ) -> None:
        """
        Args:
          num_features:
            The input dimension of the model.
          num_classes:
            The output dimension of the model.
          subsampling_factor:
            Number of output frames is num_in_frames // subsampling_factor.
            Currently, subsampling_factor MUST be 4.
          d_model:
            Attention dimension.
          nhead:
            Number of heads in multi-head attention.
            Must satisfy d_model // nhead == 0.
          dim_feedforward:
            The output dimension of the feedforward layers in encoder/decoder.
          num_encoder_layers:
            Number of encoder layers.
          num_decoder_layers:
            Number of decoder layers.
          dropout:
            Dropout in encoder/decoder.
          layer_dropout (float): layer-dropout rate.
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4 and subsampling_factor != 2:
            raise NotImplementedError("Support only 'subsampling_factor=4 or 2'.")

        # self.encoder_embed converts the input of shape (N, T, num_classes)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_classes -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder_pos = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        # TODO(fangjun): remove dropout
        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout), ScaledLinear(d_model, num_classes, bias=True)
        )

        if num_decoder_layers > 0:
            self.decoder_num_class = (
                self.num_classes
            )  # bpe model already has sos/eos symbol

            self.decoder_embed = ScaledEmbedding(
                num_embeddings=self.decoder_num_class, embedding_dim=d_model
            )
            self.decoder_pos = PositionalEncoding(d_model, dropout)

            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.decoder = TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
            )

            self.decoder_output_layer = ScaledLinear(
                d_model, self.decoder_num_class, bias=True
            )

            self.decoder_criterion = LabelSmoothingLoss()
        else:
            self.decoder_criterion = None

    def forward(
        self,
        x: torch.Tensor,
        supervision: Optional[Supervisions] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, T, C).
          supervision:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling)
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          Return a tuple containing 3 tensors:
            - CTC output for ctc decoding. Its shape is (N, T, C)
            - Encoder output with shape (T, N, C). It can be used as key and
              value for the decoder.
            - Encoder output padding mask. It can be used as
              memory_key_padding_mask for the decoder. Its shape is (N, T).
              It is None if `supervision` is None.
        """

        encoder_memory, memory_key_padding_mask = self.run_encoder(
            x, supervision, warmup
        )

        x = self.ctc_output(encoder_memory)
        return x, encoder_memory, memory_key_padding_mask

    def run_encoder(
        self,
        x: torch.Tensor,
        supervisions: Optional[Supervisions] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the transformer encoder.

        Args:
          x:
            The model input. Its shape is (N, T, C).
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute the encoder padding mask, which is used as memory key
            padding mask for the decoder.
        Returns:
          Return a tuple with two tensors:
            - The encoder output, with shape (T, N, C)
            - encoder padding mask, with shape (N, T).
              The mask is None if `supervisions` is None.
              It is used as memory key padding mask in the decoder.
        """
        x = self.encoder_embed(x)
        x = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        mask = encoder_padding_mask(x.size(0), supervisions)
        mask = mask.to(x.device) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=mask, warmup=warmup)  # (T, N, C)

        return x, mask

    def ctc_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The output tensor from the transformer encoder.
            Its shape is (T, N, C)

        Returns:
          Return a tensor that can be used for CTC decoding.
          Its shape is (N, T, C)
        """
        x = self.encoder_output_layer(x)
        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        x = nn.functional.log_softmax(x, dim=-1)  # (N, T, C)
        return x

    @torch.jit.export
    def decoder_forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[List[int]],
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        Args:
          memory:
            It's the output of the encoder with shape (T, N, C)
          memory_key_padding_mask:
            The padding mask from the encoder.
          token_ids:
            A list-of-list IDs. Each sublist contains IDs for an utterance.
            The IDs can be either phone IDs or word piece IDs.
          sos_id:
            sos token id
          eos_id:
            eos token id

        Returns:
            A scalar, the **sum** of label smoothing loss over utterances
            in the batch without any normalization.
        """
        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(ys_in, batch_first=True, padding_value=float(eos_id))

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(ys_out, batch_first=True, padding_value=float(-1))

        device = memory.device
        ys_in_pad = ys_in_pad.to(device)
        ys_out_pad = ys_out_pad.to(device)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(device)

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.decoder_embed(ys_in_pad)  # (N, T) -> (N, T, C)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (T, N, C)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        pred_pad = self.decoder_output_layer(pred_pad)  # (N, T, C)

        decoder_loss = self.decoder_criterion(pred_pad, ys_out_pad)

        return decoder_loss

    @torch.jit.export
    def decoder_nll(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[torch.Tensor],
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        Args:
          memory:
            It's the output of the encoder with shape (T, N, C)
          memory_key_padding_mask:
            The padding mask from the encoder.
          token_ids:
            A list-of-list IDs (e.g., word piece IDs).
            Each sublist represents an utterance.
          sos_id:
            The token ID for SOS.
          eos_id:
            The token ID for EOS.
        Returns:
            A 2-D tensor of shape (len(token_ids), max_token_length)
            representing the cross entropy loss (i.e., negative log-likelihood).
        """
        # The common part between this function and decoder_forward could be
        # extracted as a separate function.
        if isinstance(token_ids[0], torch.Tensor):
            # This branch is executed by torchscript in C++.
            # See https://github.com/k2-fsa/k2/pull/870
            # https://github.com/k2-fsa/k2/blob/3c1c18400060415b141ccea0115fd4bf0ad6234e/k2/torch/bin/attention_rescore.cu#L286
            token_ids = [tolist(t) for t in token_ids]

        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(ys_in, batch_first=True, padding_value=float(eos_id))

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(ys_out, batch_first=True, padding_value=float(-1))

        device = memory.device
        ys_in_pad = ys_in_pad.to(device, dtype=torch.int64)
        ys_out_pad = ys_out_pad.to(device, dtype=torch.int64)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(device)

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.decoder_embed(ys_in_pad)  # (B, T) -> (B, T, F)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (T, B, F)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, B, F) -> (B, T, F)
        pred_pad = self.decoder_output_layer(pred_pad)  # (B, T, F)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            pred_pad.view(-1, self.decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=-1,
            reduction="none",
        )

        nll = nll.view(pred_pad.shape[0], -1)

        return nll


class TransformerEncoderLayer(nn.Module):
    """
    Modified from torch.nn.TransformerEncoderLayer.

    Args:
      d_model:
        the number of expected features in the input (required).
      nhead:
        the number of heads in the multiheadattention models (required).
      dim_feedforward:
        the dimension of the feedforward network model (default=2048).
      dropout:
        the dropout value (default=0.1).
      activation:
        the activation function of intermediate layer, relu or
        gelu (default=relu).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.layer_dropout = layer_dropout

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.0)
        # Implementation of Feedforward model

        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional)
            warmup: controls selective bypass of of layers; if < 1.0, we will
              bypass layers more frequently.

        Shape:
            src: (S, N, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length,
            N is the batch size, E is the feature number
        """
        src_orig = src

        warmup_scale = min(0.1 + warmup, 1.0)
        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        # src_att = self.self_attn(src, src, src, src_mask)
        src_att = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout(src_att)

        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Modified from torch.nn.TransformerDecoderLayer.
    Add support of normalize_before,
    i.e., use layer_norm before the first block.

    Args:
      d_model:
        the number of expected features in the input (required).
      nhead:
        the number of heads in the multiheadattention models (required).
      dim_feedforward:
        the dimension of the feedforward network model (default=2048).
      dropout:
        the dropout value (default=0.1).
      activation:
        the activation function of intermediate layer, relu or
        gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        normalize_before: bool = True,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.layer_dropout = layer_dropout
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.0)
        self.src_attn = MultiheadAttention(d_model, nhead, dropout=0.0)
        # Implementation of Feedforward model
        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
          tgt:
            the sequence to the decoder layer (required).
          memory:
            the sequence from the last layer of the encoder (required).
          tgt_mask:
            the mask for the tgt sequence (optional).
          memory_mask:
            the mask for the memory sequence (optional).
          tgt_key_padding_mask:
            the mask for the tgt keys per batch (optional).
          memory_key_padding_mask:
            the mask for the memory keys per batch (optional).
          warmup: controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.



        Shape:
            tgt: (T, N, E).
            memory: (S, N, E).
            tgt_mask: (T, T).
            memory_mask: (T, S).
            tgt_key_padding_mask: (N, T).
            memory_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length,
            N is the batch size, E is the feature number
        """
        tgt_orig = tgt

        warmup_scale = min(0.1 + warmup, 1.0)
        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        # tgt_att = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt_att = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt_att)

        # src_att = self.src_attn(tgt, memory, memory, memory_mask)
        src_att = self.src_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(src_att)

        tgt = tgt + self.dropout(self.feed_forward(tgt))

        tgt = self.norm_final(self.balancer(tgt))

        if alpha != 1.0:
            tgt = alpha * tgt + (1 - alpha) * tgt_orig

        return tgt


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
            )

        return output


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(10, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        r"""Pass the input through the decoder layers in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            tgt: (S, N, E).
            tgt_mask: (S, S).
            tgt_key_padding_mask: (N, S).

        """
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                warmup=warmup,
            )

        return output


class PositionalEncoding(nn.Module):
    """This class implements the positional encoding
    proposed in the following paper:

    - Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf

        PE(pos, 2i) = sin(pos / (10000^(2i/d_modle))
        PE(pos, 2i+1) = cos(pos / (10000^(2i/d_modle))

    Note::

      1 / (10000^(2i/d_model)) = exp(-log(10000^(2i/d_model)))
                               = exp(-1* 2i / d_model * log(100000))
                               = exp(2i * -(log(10000) / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """
        Args:
          d_model:
            Embedding dimension.
          dropout:
            Dropout probability to be applied to the output of this module.
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        # not doing: self.pe = None because of errors thrown by torchscript
        self.pe = torch.zeros(1, 0, self.d_model, dtype=torch.float32)

    def extend_pe(self, x: torch.Tensor) -> None:
        """Extend the time t in the positional encoding if required.

        The shape of `self.pe` is (1, T1, d_model). The shape of the input x
        is (N, T, d_model). If T > T1, then we change the shape of self.pe
        to (N, T, d_model). Otherwise, nothing is done.

        Args:
          x:
            It is a tensor of shape (N, T, C).
        Returns:
          Return None.
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model, dtype=torch.float32)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Now pe is of shape (1, T, d_model), where T is x.size(1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
          x:
            Its shape is (N, T, C)

        Returns:
          Return a tensor of shape (N, T, C)
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Noam(object):
    """
    Implements Noam optimizer.

    Proposed in
    "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

    Modified from
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py  # noqa

    Args:
      params:
        iterable of parameters to optimize or dicts defining parameter groups
      model_size:
        attention dimension of the transformer model
      factor:
        learning rate factor
      warm_step:
        warmup steps
    """

    def __init__(
        self,
        params,
        model_size: int = 256,
        factor: float = 10.0,
        warm_step: int = 25000,
        weight_decay=0,
    ) -> None:
        """Construct an Noam object."""
        self.optimizer = torch.optim.Adam(
            params, lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay
        )
        self._step = 0
        self.warmup = warm_step
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def encoder_padding_mask(
    max_len: int,
    subsampling_factor: Optional[int] = 4,
    supervisions: Optional[Supervisions] = None,
) -> Optional[torch.Tensor]:
    """Make mask tensor containing indexes of padded part.

    TODO::
      This function **assumes** that the model uses
      a subsampling factor of 4 or 2. We should remove that
      assumption later.

    Args:
      max_len:
        Maximum length of input features.
        CAUTION: It is the length after subsampling.
      supervisions:
        Supervision in lhotse format.
        See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
        (CAUTION: It contains length information, i.e., start and number of
         frames, before subsampling)

    Returns:
        Tensor: Mask tensor of dimension (batch_size, input_length),
        True denote the masked indices.
    """
    if supervisions is None:
        return None

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"],
            supervisions["num_frames"],
        ),
        1,
    ).to(torch.int32)

    lengths = [0 for _ in range(int(supervision_segments[:, 0].max().item()) + 1)]
    for idx in range(supervision_segments.size(0)):
        # Note: TorchScript doesn't allow to unpack tensors as tuples
        sequence_idx = supervision_segments[idx, 0].item()
        start_frame = supervision_segments[idx, 1].item()
        num_frames = supervision_segments[idx, 2].item()
        lengths[sequence_idx] = start_frame + num_frames

    if subsampling_factor == 4:
        lengths = [((i - 1) // 2 - 1) // 2 for i in lengths]
    elif subsampling_factor == 2:
        lengths = [(i - 1) // 2 - 2 for i in lengths]
    bs = int(len(lengths))
    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    # Note: TorchScript doesn't implement Tensor.new()
    seq_length_expand = torch.tensor(
        lengths, device=seq_range_expand.device, dtype=seq_range_expand.dtype
    ).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def decoder_padding_mask(ys_pad: torch.Tensor, ignore_id: int = -1) -> torch.Tensor:
    """Generate a length mask for input.

    The masked position are filled with True,
    Unmasked positions are filled with False.

    Args:
      ys_pad:
        padded tensor of dimension (batch_size, input_length).
      ignore_id:
        the ignored number (the padding number) in ys_pad

    Returns:
      Tensor:
        a bool tensor of the same shape as the input tensor.
    """
    ys_mask = ys_pad == ignore_id
    return ys_mask


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are
    filled with float('-inf'). Unmasked positions are filled with float(0.0).
    The mask can be used for masked self-attention.

    For instance, if sz is 3, it returns::

        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0]])

    Args:
      sz: mask size

    Returns:
      A square mask of dimension (sz, sz)
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def add_sos(token_ids: List[List[int]], sos_id: int) -> List[List[int]]:
    """Prepend sos_id to each utterance.

    Args:
      token_ids:
        A list-of-list of token IDs. Each sublist contains
        token IDs (e.g., word piece IDs) of an utterance.
      sos_id:
        The ID of the SOS token.

    Return:
      Return a new list-of-list, where each sublist starts
      with SOS ID.
    """
    return [[sos_id] + utt for utt in token_ids]


def add_eos(token_ids: List[List[int]], eos_id: int) -> List[List[int]]:
    """Append eos_id to each utterance.

    Args:
      token_ids:
        A list-of-list of token IDs. Each sublist contains
        token IDs (e.g., word piece IDs) of an utterance.
      eos_id:
        The ID of the EOS token.

    Return:
      Return a new list-of-list, where each sublist ends
      with EOS ID.
    """
    return [utt + [eos_id] for utt in token_ids]


def tolist(t: torch.Tensor) -> List[int]:
    """Used by jit"""
    return torch.jit.annotate(List[int], t.tolist())
