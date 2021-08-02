# Copyright (c) 2021 University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import math
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from subsampling import Conv2dSubsampling, VggSubsampling

from icefall.utils import get_texts

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
        normalize_before: bool = True,
        vgg_frontend: bool = False,
        mmi_loss: bool = True,
        use_feat_batchnorm: bool = False,
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
          normalize_before:
            If True, use pre-layer norm; False to use post-layer norm.
          vgg_frontend:
            True to use vgg style frontend for subsampling.
          mmi_loss:
          use_feat_batchnorm:
            True to use batchnorm for the input layer.
        """
        super().__init__()
        self.use_feat_batchnorm = use_feat_batchnorm
        if use_feat_batchnorm:
            self.feat_batchnorm = nn.BatchNorm1d(num_features)

        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape [N, T, num_classes]
        # to the shape [N, T//subsampling_factor, d_model].
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_classes -> d_model
        if vgg_frontend:
            self.encoder_embed = VggSubsampling(num_features, d_model)
        else:
            self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder_pos = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=normalize_before,
        )

        if normalize_before:
            encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm = None

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )

        # TODO(fangjun): remove dropout
        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(d_model, num_classes)
        )

        if num_decoder_layers > 0:
            if mmi_loss:
                self.decoder_num_class = (
                    self.num_classes + 1
                )  # +1 for the sos/eos symbol
            else:
                self.decoder_num_class = (
                    self.num_classes
                )  # bpe model already has sos/eos symbol

            self.decoder_embed = nn.Embedding(
                num_embeddings=self.decoder_num_class, embedding_dim=d_model
            )
            self.decoder_pos = PositionalEncoding(d_model, dropout)

            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=normalize_before,
            )

            if normalize_before:
                decoder_norm = nn.LayerNorm(d_model)
            else:
                decoder_norm = None

            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
                norm=decoder_norm,
            )

            self.decoder_output_layer = torch.nn.Linear(
                d_model, self.decoder_num_class
            )

            self.decoder_criterion = LabelSmoothingLoss(self.decoder_num_class)
        else:
            self.decoder_criterion = None

    def forward(
        self, x: torch.Tensor, supervision: Optional[Supervisions] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is [N, T, C].
          supervision:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling)

        Returns:
          Return a tuple containing 3 tensors:
            - CTC output for ctc decoding. Its shape is [N, T, C]
            - Encoder output with shape [T, N, C]. It can be used as key and
              value for the decoder.
            - Encoder output padding mask. It can be used as
              memory_key_padding_mask for the decoder. Its shape is [N, T].
              It is None if `supervision` is None.
        """
        if self.use_feat_batchnorm:
            x = x.permute(0, 2, 1)  # [N, T, C] -> [N, C, T]
            x = self.feat_batchnorm(x)
            x = x.permute(0, 2, 1)  # [N, C, T] -> [N, T, C]
        encoder_memory, memory_key_padding_mask = self.encode(x, supervision)
        x = self.encoder_output(encoder_memory)
        return x, encoder_memory, memory_key_padding_mask

    def encode(
        self, x: torch.Tensor, supervisions: Optional[Supervisions] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            The model input. Its shape is [N, T, C].
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.
        Returns:
          Return a tuple with two tensors:
            - The encoder output, with shape [T, N, C]
            - encoder padding mask, with shape [N, T].
              The mask is None if `supervisions` is None.
              It is used as memory key padding mask in the decoder.
        """
        x = self.encoder_embed(x)
        x = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        mask = encoder_padding_mask(x.size(0), supervisions)
        mask = mask.to(x.device) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=mask)  # (T, N, C)

        return x, mask

    def encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The output tensor from the transformer encoder.
            Its shape is [T, N, C]

        Returns:
          Return a tensor that can be used for CTC decoding.
          Its shape is [N, T, C]
        """
        x = self.encoder_output_layer(x)
        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        x = nn.functional.log_softmax(x, dim=-1)  # (N, T, C)
        return x

    def decoder_forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        supervision: Optional[Supervisions] = None,
        L_inv: Optional[k2.Fsa] = None,
        word_table: Optional[k2.SymbolTable] = None,
        oov_str: Optional[str] = None,
        token_ids: List[List[int]] = None,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Note:
          If phone based lexicon is used, the following arguments are required:

            - supervision
            - L_inv
            - word_table
            - oov_str

          If BPE based lexicon is used, the following arguments are required:

            - token_ids
            - sos_id
            - eos_id

        Args:
          memory:
            It's the output of the encoder with shape [T, N, C]
          memory_key_padding_mask:
            The padding mask from the encoder.
          supervision:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling)
          L_inv:
            It is an FSA with labels being word IDs and aux_labels being
            token IDs (e.g., phone IDs or word piece IDs).
          word_table:
            Word table providing mapping between words and IDs.
          oov_str:
            The OOV word, e.g., '<UNK>'
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
        if supervision is not None and word_table is not None:
            batch_text = get_normal_transcripts(
                supervision, word_table, oov_str
            )
            ys_in_pad, ys_out_pad = add_sos_eos(
                batch_text,
                L_inv,
                sos_id,
                eos_id,
            )
        elif token_ids is not None:
            _sos = torch.tensor([sos_id])
            _eos = torch.tensor([eos_id])
            ys_in = [
                torch.cat([_sos, torch.tensor(y)], dim=0) for y in token_ids
            ]
            ys_out = [
                torch.cat([torch.tensor(y), _eos], dim=0) for y in token_ids
            ]
            ys_in_pad = pad_list(ys_in, eos_id)
            ys_out_pad = pad_list(ys_out, -1)
        else:
            raise ValueError("Invalid input for decoder self attention")

        device = memory.device
        ys_in_pad = ys_in_pad.to(device)
        ys_out_pad = ys_out_pad.to(device)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad)

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

    def decoder_nll(
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
            It's the output of the encoder with shape [T, N, C]
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
        if token_ids is not None:
            _sos = torch.tensor([sos_id])
            _eos = torch.tensor([eos_id])
            ys_in = [
                torch.cat([_sos, torch.tensor(y)], dim=0) for y in token_ids
            ]
            ys_out = [
                torch.cat([torch.tensor(y), _eos], dim=0) for y in token_ids
            ]
            ys_in_pad = pad_list(ys_in, eos_id)
            ys_out_pad = pad_list(ys_out, -1)
        else:
            raise ValueError("Invalid input for decoder self attention")

        device = memory.device
        ys_in_pad = ys_in_pad.to(device, dtype=torch.int64)
        ys_out_pad = ys_out_pad.to(device, dtype=torch.int64)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad)

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
      normalize_before:
        whether to use layer_norm before the first block.

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
        activation: str = "relu",
        normalize_before: bool = True,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional)

        Shape:
            src: (S, N, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length,
            N is the batch size, E is the feature number
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.normalize_before:
            src = self.norm2(src)
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
        activation: str = "relu",
        normalize_before: bool = True,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.src_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = nn.functional.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
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
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = residual + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt2 = self.src_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = residual + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation: str):
    if activation == "relu":
        return nn.functional.relu
    elif activation == "gelu":
        return nn.functional.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )


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
        self.pe = None

    def extend_pe(self, x: torch.Tensor) -> None:
        """Extend the time t in the positional encoding if required.

        The shape of `self.pe` is [1, T1, d_model]. The shape of the input x
        is [N, T, d_model]. If T > T1, then we change the shape of self.pe
        to [N, T, d_model]. Otherwise, nothing is done.

        Args:
          x:
            It is a tensor of shape [N, T, C].
        Returns:
          Return None.
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
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
        # Now pe is of shape [1, T, d_model], where T is x.size(1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
          x:
            Its shape is [N, T, C]

        Returns:
          Return a tensor of shape [N, T, C]
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


class LabelSmoothingLoss(nn.Module):
    """
    Label-smoothing loss. KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    Modified from
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py  # noqa

    Args:
        size: the number of class
        padding_idx: padding_idx: ignored class id
        smoothing: smoothing rate (0.0 means the conventional CE)
        normalize_length: normalize loss by sequence length if True
        criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size: int,
        padding_idx: int = -1,
        smoothing: float = 0.1,
        normalize_length: bool = False,
        criterion: nn.Module = nn.KLDivLoss(reduction="none"),
    ) -> None:
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        assert 0.0 < smoothing <= 1.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between x and target.

        Args:
          x:
            prediction of dimension
            (batch_size, input_length, number_of_classes).
          target:
            target masked with self.padding_id of
            dimension (batch_size, input_length).

        Returns:
          A scalar tensor containing the loss without normalization.
        """
        assert x.size(2) == self.size
        #  batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        #  denom = total if self.normalize_length else batch_size
        denom = total if self.normalize_length else 1
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


def encoder_padding_mask(
    max_len: int, supervisions: Optional[Supervisions] = None
) -> Optional[torch.Tensor]:
    """Make mask tensor containing indexes of padded part.

    TODO::
      This function **assumes** that the model uses
      a subsampling factor of 4. We should remove that
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
        Tensor: Mask tensor of dimension (batch_size, input_length), True denote the masked indices.
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

    lengths = [
        0 for _ in range(int(supervision_segments[:, 0].max().item()) + 1)
    ]
    for idx in range(supervision_segments.size(0)):
        # Note: TorchScript doesn't allow to unpack tensors as tuples
        sequence_idx = supervision_segments[idx, 0].item()
        start_frame = supervision_segments[idx, 1].item()
        num_frames = supervision_segments[idx, 2].item()
        lengths[sequence_idx] = start_frame + num_frames

    lengths = [((i - 1) // 2 - 1) // 2 for i in lengths]
    bs = int(len(lengths))
    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    # Note: TorchScript doesn't implement Tensor.new()
    seq_length_expand = torch.tensor(
        lengths, device=seq_range_expand.device, dtype=seq_range_expand.dtype
    ).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def decoder_padding_mask(
    ys_pad: torch.Tensor, ignore_id: int = -1
) -> torch.Tensor:
    """Generate a length mask for input.

    The masked position are filled with bool(True),
    Unmasked positions are filled with bool(False).

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


def get_normal_transcripts(
    supervision: Supervisions, words: k2.SymbolTable, oov: str = "<UNK>"
) -> List[List[int]]:
    """Get normal transcripts (1 input recording has 1 transcript)
    from lhotse cut format.

    Achieved by concatenating the transcripts corresponding to the
    same recording.

    Args:
      supervision:
        Supervision in lhotse format.
        See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
      words:
        The word symbol table.
      oov:
        Out of vocabulary word.

    Returns:
        List[List[int]]: List of concatenated transcripts, length is batch_size
    """

    texts = [
        [token if token in words else oov for token in text.split(" ")]
        for text in supervision["text"]
    ]
    texts_ids = [[words[token] for token in text] for text in texts]

    batch_text = [
        [] for _ in range(int(supervision["sequence_idx"].max().item()) + 1)
    ]
    for sequence_idx, text in zip(supervision["sequence_idx"], texts_ids):
        batch_text[sequence_idx] = batch_text[sequence_idx] + text
    return batch_text


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are
    filled with float('-inf'). Unmasked positions are filled with float(0.0).

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


def add_sos_eos(
    ys: List[List[int]],
    L_inv: k2.Fsa,
    sos_id: int,
    eos_id: int,
    ignore_id: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
      ys:
        Batch of unpadded target sequences (i.e., word IDs)
      L_inv:
        Its labels are words, while its aux_labels are tokens.
      sos_id:
        index of <sos>
      eos_id:
        index of <eos>
      ignore_id:
        value for padding

    Returns:
      Return a tuple containing two tensors:
       - Input of transformer decoder.
         Padded tensor of dimension (batch_size, max_length).
       - Output of transformer decoder.
         Padded tensor of dimension (batch_size, max_length).
    """

    _sos = torch.tensor([sos_id])
    _eos = torch.tensor([eos_id])
    ys = get_hierarchical_targets(ys, L_inv)
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos_id), pad_list(ys_out, ignore_id)


def pad_list(ys: List[torch.Tensor], pad_value: float) -> torch.Tensor:
    """Perform padding for the list of tensors.

    Args:
      ys:
        List of tensors. len(ys) = batch_size.
      pad_value:
        Value for padding.

    Returns:
        Tensor: Padded tensor (batch_size, max_length, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(ys)
    max_len = max(x.size(0) for x in ys)
    pad = ys[0].new_full((n_batch, max_len, *ys[0].size()[1:]), pad_value)

    for i in range(n_batch):
        pad[i, : ys[i].size(0)] = ys[i]

    return pad


def get_hierarchical_targets(
    ys: List[List[int]], L_inv: Optional[k2.Fsa] = None
) -> List[torch.Tensor]:
    """Get hierarchical transcripts (i.e., phone level transcripts) from
    transcripts (i.e., word level transcripts).

    Args:
      ys:
        Word level transcripts. Each sublist is a transcript of an utterance.
      L_inv:
        Its labels are words, while its aux_labels are tokens.

    Returns:
      List[torch.Tensor]:
        Token level transcripts.
    """

    if L_inv is None:
        return [torch.tensor(y) for y in ys]

    device = L_inv.device

    transcripts = k2.create_fsa_vec(
        [k2.linear_fsa(x, device=device) for x in ys]
    )
    transcripts_with_self_loops = k2.add_epsilon_self_loops(transcripts)

    transcripts_lexicon = k2.intersect(
        L_inv, transcripts_with_self_loops, treat_epsilons_specially=False
    )
    # Don't call invert_() above because we want to return phone IDs,
    # which is the `aux_labels` of transcripts_lexicon
    transcripts_lexicon = k2.remove_epsilon(transcripts_lexicon)
    transcripts_lexicon = k2.top_sort(transcripts_lexicon)

    transcripts_lexicon = k2.shortest_path(
        transcripts_lexicon, use_double_scores=True
    )

    ys = get_texts(transcripts_lexicon)
    ys = [torch.tensor(y) for y in ys]

    return ys
