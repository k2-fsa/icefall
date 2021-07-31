#!/usr/bin/env python3

# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import math
from typing import Dict, List, Optional, Tuple

import k2
import torch
from torch import Tensor, nn

from icefall.utils import get_texts

# Note: TorchScript requires Dict/List/etc. to be fully typed.
Supervisions = Dict[str, Tensor]


class Transformer(nn.Module):
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
        normalize_before: bool = True,
        vgg_frontend: bool = False,
        mmi_loss: bool = True,
        use_feat_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.use_feat_batchnorm = use_feat_batchnorm
        if use_feat_batchnorm:
            self.feat_batchnorm = nn.BatchNorm1d(num_features)

        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        self.encoder_embed = (
            VggSubsampling(num_features, d_model)
            if vgg_frontend
            else Conv2dSubsampling(num_features, d_model)
        )
        self.encoder_pos = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            normalize_before=normalize_before,
        )

        if normalize_before:
            encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm = None

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

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

            self.decoder_embed = nn.Embedding(self.decoder_num_class, d_model)
            self.decoder_pos = PositionalEncoding(d_model, dropout)

            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                normalize_before=normalize_before,
            )

            if normalize_before:
                decoder_norm = nn.LayerNorm(d_model)
            else:
                decoder_norm = None

            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

            self.decoder_output_layer = torch.nn.Linear(
                d_model, self.decoder_num_class
            )

            self.decoder_criterion = LabelSmoothingLoss(self.decoder_num_class)
        else:
            self.decoder_criterion = None

    def forward(
        self, x: Tensor, supervision: Optional[Supervisions] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            x: Tensor of dimension (batch_size, num_features, input_length).
            supervision: Supervison in lhotse format, get from batch['supervisions']

        Returns:
            Tensor: After log-softmax tensor of dimension (batch_size, number_of_classes, input_length).
            Tensor: Before linear layer tensor of dimension (input_length, batch_size, d_model).
            Optional[Tensor]: Mask tensor of dimension (batch_size, input_length) or None.

        """
        if self.use_feat_batchnorm:
            x = self.feat_batchnorm(x)
        encoder_memory, memory_mask = self.encode(x, supervision)
        x = self.encoder_output(encoder_memory)
        return x, encoder_memory, memory_mask

    def encode(
        self, x: Tensor, supervisions: Optional[Supervisions] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: Tensor of dimension (batch_size, num_features, input_length).
            supervisions : Supervison in lhotse format, i.e., batch['supervisions']

        Returns:
            Tensor: Predictor tensor of dimension (input_length, batch_size, d_model).
            Optional[Tensor]: Mask tensor of dimension (batch_size, input_length) or None.
        """
        x = x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)

        x = self.encoder_embed(x)
        x = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        mask = encoder_padding_mask(x.size(0), supervisions)
        mask = mask.to(x.device) if mask != None else None
        x = self.encoder(x, src_key_padding_mask=mask)  # (T, B, F)

        return x, mask

    def encoder_output(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of dimension (input_length, batch_size, d_model).

        Returns:
            Tensor: After log-softmax tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.encoder_output_layer(x).permute(
            1, 2, 0
        )  # (T, B, F) ->(B, F, T)
        x = nn.functional.log_softmax(x, dim=1)  # (B, F, T)
        return x

    def decoder_forward(
        self,
        x: Tensor,
        encoder_mask: Tensor,
        supervision: Supervisions = None,
        graph_compiler: object = None,
        token_ids: List[List[int]] = None,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            x: Tensor of dimension (input_length, batch_size, d_model).
            encoder_mask: Mask tensor of dimension (batch_size, input_length)
            supervision: Supervison in lhotse format, get from batch['supervisions']
            graph_compiler: use graph_compiler.L_inv (Its labels are words, while its aux_labels are phones)
                            , graph_compiler.words and graph_compiler.oov
            token_ids: A list of lists. Each list contains word piece IDs for an utterance.
            sos_id: sos token id
            eos_id: eos token id

        Returns:
            Tensor: Decoder loss.
        """
        if supervision is not None and graph_compiler is not None:
            batch_text = get_normal_transcripts(
                supervision, graph_compiler.lexicon.words, graph_compiler.oov
            )
            ys_in_pad, ys_out_pad = add_sos_eos(
                batch_text,
                graph_compiler.L_inv,
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

        ys_in_pad = ys_in_pad.to(x.device)
        ys_out_pad = ys_out_pad.to(x.device)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            x.device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad)

        tgt = self.decoder_embed(ys_in_pad)  # (B, T) -> (B, T, F)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_mask,
        )  # (T, B, F)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, B, F) -> (B, T, F)
        pred_pad = self.decoder_output_layer(pred_pad)  # (B, T, F)

        decoder_loss = self.decoder_criterion(pred_pad, ys_out_pad)

        return decoder_loss

    def decoder_nll(
        self,
        x: Tensor,
        encoder_mask: Tensor,
        token_ids: List[List[int]],
        sos_id: int,
        eos_id: int,
    ) -> Tensor:
        """
        Args:
            x: encoder-output, Tensor of dimension (input_length, batch_size, d_model).
            encoder_mask: Mask tensor of dimension (batch_size, input_length)
            token_ids: n-best list extracted from lattice before rescore

        Returns:
            Tensor: negative log-likelihood.
        """
        # The common part between this fuction and decoder_forward could be
        # extracted as a seperated function.
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

        ys_in_pad = ys_in_pad.to(x.device, dtype=torch.int64)
        ys_out_pad = ys_out_pad.to(x.device, dtype=torch.int64)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            x.device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad)

        tgt = self.decoder_embed(ys_in_pad)  # (B, T) -> (B, T, F)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_mask,
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
    Modified from torch.nn.TransformerEncoderLayer. Add support of normalize_before,
    i.e., use layer_norm before the first block.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        normalize_before: whether to use layer_norm before the first block.

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
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
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
    Modified from torch.nn.TransformerDecoderLayer. Add support of normalize_before,
    i.e., use layer_norm before the first block.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

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
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            tgt: (T, N, E).
            memory: (S, N, E).
            tgt_mask: (T, T).
            memory_mask: (T, S).
            tgt_key_padding_mask: (N, T).
            memory_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
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


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
        Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py

    Args:
        idim: Input dimension.
        odim: Output dimension.

    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct a Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=odim, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=odim, out_channels=odim, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )
        self.out = nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: Tensor) -> Tensor:
        """Subsample x.

        Args:
            x: Input tensor of dimension (batch_size, input_length, num_features). (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor of dimension (batch_size, input_length, d_model).
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x


class VggSubsampling(nn.Module):
    """Trying to follow the setup described here https://arxiv.org/pdf/1910.09799.pdf
       This paper is not 100% explicit so I am guessing to some extent,
       and trying to compare with other VGG implementations.

    Args:
        idim: Input dimension.
        odim: Output dimension.

    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct a VggSubsampling object.   This uses 2 VGG blocks with 2
        Conv2d layers each, subsampling its input by a factor of 4 in the
        time dimensions.

        Args:
          idim:  Number of features at input, e.g. 40 or 80 for MFCC
                 (will be treated as the image height).
          odim:  Output dimension (number of features), e.g. 256
        """
        super(VggSubsampling, self).__init__()

        cur_channels = 1
        layers = []
        block_dims = [32, 64]

        # The decision to use padding=1 for the 1st convolution, then padding=0
        # for the 2nd and for the max-pooling, and ceil_mode=True, was driven by
        # a back-compatibility concern so that the number of frames at the
        # output would be equal to:
        #  (((T-1)//2)-1)//2.
        # We can consider changing this by using padding=1 on the 2nd convolution,
        # so the num-frames at the output would be T//4.
        for block_dim in block_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Conv2d(
                    in_channels=block_dim,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                )
            )
            layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=True
                )
            )
            cur_channels = block_dim

        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(
            block_dims[-1] * (((idim - 1) // 2 - 1) // 2), odim
        )

    def forward(self, x: Tensor) -> Tensor:
        """Subsample x.

        Args:
            x: Input tensor of dimension (batch_size, input_length, num_features). (#batch, time, idim).

        Returns:
           torch.Tensor: Subsampled tensor of dimension (batch_size, input_length', d_model).
              where input_length' == (((input_length - 1) // 2) - 1) // 2

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.layers(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding.

        Args:
            x: Input tensor of dimention (batch_size, input_length, d_model).

        Returns:
            torch.Tensor: Encoded tensor of dimention (batch_size, input_length, d_model).

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Noam(object):
    """
    Implements Noam optimizer. Proposed in "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        model_size: attention dimension of the transformer model
        factor: learning rate factor
        warm_step: warmup steps
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
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py

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

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """
        Compute loss between x and target.

        Args:
            x: prediction of dimention (batch_size, input_length, number_of_classes).
            target: target masked with self.padding_id of dimention (batch_size, input_length).

        Returns:
            torch.Tensor: scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
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
) -> Optional[Tensor]:
    """Make mask tensor containing indices of padded part.

    Args:
        max_len: maximum length of input features
        supervisions : Supervison in lhotse format, i.e., batch['supervisions']

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


def decoder_padding_mask(ys_pad: Tensor, ignore_id: int = -1) -> Tensor:
    """Generate a length mask for input. The masked position are filled with bool(True),
        Unmasked positions are filled with bool(False).

    Args:
        ys_pad: padded tensor of dimension (batch_size, input_length).
        ignore_id: the ignored number (the padding number) in ys_pad

    Returns:
        Tensor: a mask tensor of dimension (batch_size, input_length).
    """
    ys_mask = ys_pad == ignore_id
    return ys_mask


def get_normal_transcripts(
    supervision: Supervisions, words: k2.SymbolTable, oov: str = "<UNK>"
) -> List[List[int]]:
    """Get normal transcripts (1 input recording has 1 transcript) from lhotse cut format.
    Achieved by concatenate the transcripts corresponding to the same recording.

    Args:
        supervision : Supervison in lhotse format, i.e., batch['supervisions']
        words: The word symbol table.
        oov: Out of vocabulary word.

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


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

    Args:
        sz: mask size

    Returns:
        Tensor: a square mask of dimension (sz, sz)
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
    lexicon: k2.Fsa,
    sos_id: int,
    eos_id: int,
    ignore_id: int = -1,
) -> Tuple[Tensor, Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys: batch of unpadded target sequences
        lexicon: Its labels are words, while its aux_labels are phones.
        sos_id: index of <sos>
        eos_id: index of <eos>
        ignore_id: index of padding

    Returns:
        Tensor: Input of transformer decoder. Padded tensor of dimention (batch_size, max_length).
        Tensor: Output of transformer decoder. padded tensor of dimention (batch_size, max_length).
    """

    _sos = torch.tensor([sos_id])
    _eos = torch.tensor([eos_id])
    ys = get_hierarchical_targets(ys, lexicon)
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def pad_list(ys: List[Tensor], pad_value: float) -> Tensor:
    """Perform padding for the list of tensors.

    Args:
        ys: List of tensors. len(ys) = batch_size.
        pad_value: Value for padding.

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
    ys: List[List[int]], lexicon: k2.Fsa
) -> List[Tensor]:
    """Get hierarchical transcripts (i.e., phone level transcripts) from transcripts (i.e., word level transcripts).

    Args:
        ys: Word level transcripts.
        lexicon: Its labels are words, while its aux_labels are phones.

    Returns:
        List[Tensor]: Phone level transcripts.

    """

    if lexicon is None:
        return ys
    else:
        L_inv = lexicon

    n_batch = len(ys)
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


def test_transformer():
    t = Transformer(40, 1281)
    T = 200
    f = torch.rand(31, 40, T)
    g, _, _ = t(f)
    assert g.shape == (31, 1281, (((T - 1) // 2) - 1) // 2)


def main():
    test_transformer()


if __name__ == "__main__":
    main()
