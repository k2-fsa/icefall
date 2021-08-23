# Copyright (c) 2021 University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Note: TorchScript requires Dict/List/etc. to be fully typed.
Supervisions = Dict[str, torch.Tensor]


class MaskedLmConformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
    ) -> None:
        """
        Args:
          num_classes:
            The input and output dimension of the model (inputs and outputs are
            both discrete)
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
       """
        super(MaskedLmConformer, self).__init__()

        self.num_classes = num_classes

        # self.embed is the embedding used for both the encoder and decoder.
        self.embed_scale = d_model ** 0.5
        self.embed = nn.Embedding(
            num_embeddings=self.decoder_num_class, embedding_dim=d_model,
            _weight=torch.randn(self.decoder_num_class, d_model) * (1 / self.embed_scale)
        )

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = MaskedLmConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            cnn_module_kernel,
        )
        self.encoder = MaskedLmConformerEncoder(encoder_layer, num_encoder_layers,
                                                norm=nn.LayerNorm(d_model))

        if num_decoder_layers > 0:
            self.decoder_num_class = self.num_classes

            decoder_layer = TransformerDecoderLayerRelPos(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            # Projects the embedding of `src`, to be added to `memory`
            self.src_linear = torch.nn.Linear(d_model, d_model)

            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoderRelPos(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
                norm=decoder_norm,
            )

            self.decoder_output_layer = torch.nn.Linear(
                d_model, self.decoder_num_class
            )


    def forward(
            self,
            masked_src_symbols: torch.Tensor,
            key_padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          masked_src_symbols:
             The input symbols to be embedded (will actually have query positions
             masked), as a Tensor of shape (batch_size, seq_len) and dtype=torch.int64.
             I.e. shape (N, T)
          key_padding_mask:
             Either None, or a Tensor of shape (batch_size, seq_len) i.e. (N, T),
             and dtype=torch.bool which has True in positions to be masked in attention
             layers and convolutions because they represent padding at the ends of
             sequences.


        Returns:
          Returns (encoded, pos_emb), where:
            `encoded` is a Tensor containing the encoded data; it is of shape (N, T, C)
                where C is the embedding_dim.
            `pos_emb` is a Tensor containing the relative positional encoding, of
                      shape (1, 2*T-1, C)
        """

        x = self.embed(masked_src_symbols) * self.embed_scale # (N, T, C)
        x, pos_emb = self.encoder_pos(x)  # pos_emb: (1, 2*T-1, C)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        x = self.encoder(x, pos_emb, key_padding_mask=key_padding_mask)  # (T, N, C)

        return x, pos_emb

    def decoder_nll(
        self,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        src_symbols: torch.Tensor,
        tgt_symbols: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          memory:
            The output of the encoder, with shape (T, N, C)
          pos_emb:
            Relative positional embedding, of shape (1, 2*T-1, C), as
            returned from the encoder
          src_symbols:
            The un-masked src symbols, a LongTensor of shape (N, T).
            Can be used to predict the target
            only in a left-to-right manner (otherwise it's cheating).
          tgt_symbols:
            Target symbols, a LongTensor of shape (N, T).
            The same as src_symbols, but shifted by one (and also,
            without symbol randomization, see randomize_proportion
            in dataloader)
          key_padding_mask:
            A BoolTensor of shape (N, T), with True for positions
            that correspond to padding at the end of source and
            memory sequences.  The same mask is used for self-attention
            and cross-attention, since the padding is the same.

        Returns:
            Returns a tensor of shape (N, T), containing the negative
            log-probabilities for the target symbols at each position
            in the target sequence.
        """
        (T, N, C) = memory.shape

        tgt_mask = generate_square_subsequent_mask(T, memory.device)

        src = self.embed(src_symbols) * self.embed_scale # (N, T) -> (N, T, C)
        src = src.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)


        src = memory + self.src_linear(src)   # (T, N, C)

        # This is a little confusing, how "tgt" is set to src.  "src" is the
        # symbol sequence without masking but with padding and randomization.
        # "tgt" is like "src" but shifted by one.
        pred = self.decoder(
            tgt=src,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )  # (T, N, C)

        pred = pred_pad.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        pred = self.decoder_output_layer(pred)  # (N, T, C)

        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            pred.view(-1, self.decoder_num_class),
            tgt_symbols.view(-1),
            reduction="none",
        )
        nll = nll.view(N, T)
        return nll




class TransformerDecoderRelPos(Module):
    r"""TransformerDecoderRelPos is a stack of N decoder layers.
      This is modified from nn.TransformerDecoder to support relative positional
      encoding.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayerRelPos() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayerRelPos(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoderRelPos(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> pos_enc = torch.rand()
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x: Tensor,
                pos_emb: Tensor,
                memory: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            x: the input embedding sequence to the decoder (required): shape = (T, N, C).
                Will be an embedding of `src_symbols` in practice
          pos_emb:
             A torch.Tensor with dtype=torch.float and shape (1, 2*T-1, C) with c==num_channels,
             representing the relative positional encoding.
          memory: the sequence from the last layer of the encoder (required):
                 shape = (T, N, C)
          attn_mask: the mask for the `x` sequence's attention to itself,
                 of shape (T, T); in practice, will ensure that no
                 position can attend to later positions.  A torch.Tensor  with dtype=torch.float
                 or dtype=torch.bool.
          key_padding_mask: the key-padding mask for both the memory and x sequences,
              a torch.Tensor with dtype=bool and shape (N, T): true for masked
              positions after the ends of sequences.
        """

        for mod in self.layers:
            x = mod(x, pos_emb, memory, x_mask=x_mask,
                    key_padding_mask=key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayerRelPos(nn.Module):
    """
    Modified from torch.nn.TransformerDecoderLayer.
    Add it to use normalize_before (hardcoded to True), i.e. use layer_norm before the first block;
    to use relative positional encoding; and for some changes/simplifications in interface
     because both sequences are the same length and have the same mask.

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
        >>> decoder_layer = nn.TransformerDecoderLayerRelPos(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> pos_emb = torch.rand(1, 20*2+1, 512)
        >>> out = decoder_layer(tgt, pos_emb, memory)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)
        self.src_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)
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


    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = nn.functional.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        memory: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
          x
            The input embedding, to be added to by the forward function, of shape (T, N, C).
            Attention within x will be left-to-right only (causal), thanks to x_mask.
          pos_emb:
            A torch.Tensor with dtype=torch.float and shape (1, 2*T-1, C) with c==num_channels,
            containing the relative positional encoding.
          memory:
            the sequence from the last layer of the encoder (required).  Shape = (T, N, C)
          x_mask:
            the mask for the x, to enforce causal (left to right) attention (optional).
            Shape == (T, T); may be bool or float.  The first T pertains to the output,
            the second T to the input.
          key_padding_mask:
            the key-padding mask to use for both the x and memory sequences.  Shep == (N, T);
            may be bool (True==masked) or float (to be added to attention scores).

        Returns:
            Returns 'x plus something', a torch.Tensor with dtype the same as x (e.g. float),
            and shape (T, N, C).
        """
        residual = x
        x = self.norm1(x)
        self_attn = self.self_attn(x, x, x,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False
                                   attn_mask=x_mask,
        )[0]
        x = residual + self.dropout1(self_attn)

        residual = x
        x = self.norm2(x)
        src_attn = self.src_attn(x, memory, memory,
                                 key_padding_mask=key_padding_mask,
                                 need_weights=False,
        )[0]
        x = residual + self.dropout2(src_attn)

        residual = x
        x = self.norm3(x)
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout3(ff)
        return x


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




def decoder_padding_mask(
    ys_pad: torch.Tensor, ignore_id: int = -1
) -> torch.Tensor:
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


def generate_square_subsequent_mask(sz: int, device: torch.Device) -> torch.Tensor:
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
    mask = (torch.triu(torch.ones(sz, sz, device=torch.Device)) == 1).transpose(0, 1)
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
    ans = []
    for utt in token_ids:
        ans.append([sos_id] + utt)
    return ans


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
    ans = []
    for utt in token_ids:
        ans.append(utt + [eos_id])
    return ans



class MaskedConvolutionModule(nn.Module):
    """
    This is used in the MaskedLmConformerLayer.  It is the same as the ConvolutionModule
    of theConformer code, but with key_padding_mask supported to make the output independent
    of the batching.

    Modified, ultimately, from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).
    """

    def __init__(
        self, channels: int, kernel_size: int, bias: bool = True
    ) -> None:
        """Construct a MaskedConvolutionModule object."""
        super(MaskedConvolutionModule, self).__init__()
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
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (T, N, C) == (#time, batch, channels).
            key_padding_mask: if supplied, a Tensor with dtype=torch.Bool and
                  shape (N, T), with True for positions that correspond to
                  padding (and should be zeroed in convolutions).

        Returns:
            Tensor: Output tensor (T, N, C)

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # Logical-not key_padding_mask, unsqueeze to shape (N, 1, T) and convert
        # to float.  Then we can just multiply by it when we need to apply
        # masking, i.e. prior to the convolution over time.
        if key_padding_mask is not None:
            x = x * torch.logical_not(key_padding_mask).unsqueeze(1).to(dtype=x.dtype)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)  # (time, batch, channel)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)



class MaskedLmConformerEncoderLayer(nn.Module):
    """
    MaskedLmConformerEncoderLayer is made up of self-attn, feedforward and convolution
    networks.  It's a simplified version of the conformer code we were previously
    using, with pre-normalization hard-coded, relative positional encoding,
    LayerNorm instead of BatchNorm in the convolution layers, and the key_padding_mask
    applied also in the convolution layers.

    See: "Conformer: Convolution-augmented Transformer for Speech Recognition", for
    the basic conformer.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
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
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = RelPositionMultiheadAttention(
            d_model, nhead, dropout=0.0
        )

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

        self.conv_module = MaskedConvolutionModule(d_model, cnn_module_kernel)

        self.norm_ff_macaron = nn.LayerNorm(
            d_model
        )  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(
            d_model
        )  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            x: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            attn_mask: the mask for the x sequence's attention to itself (optional);
                   of shape (T, T)
            key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            x: (T, N, C) i.e. (seq_len, batch_size, num_channels)
            pos_emb: (N, 2*T-1, C)
            attn_mask: (T, T) or (N*num_heads, T, T), of dtype torch.bool or torch.float, where
                  the 1st S is interpreted as the target sequence (output) and the 2nd as the source
                  sequence (input).
            key_padding_mask: (N, T), of dtype torch.bool

            T is the sequence length, N is the batch size, C is the number of channels.
        Return:
            Returns x with something added to it, of shape (T, N, C)
        """

        # macaron style feed forward module
        residual = x
        x = self.norm_ff_macaron(x)
        x = residual + self.ff_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        # multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)
        self_attn = self.self_attn(x, x, x,
                                   pos_emb=pos_emb,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False
        )[0]
        x = residual + self.dropout(self_attn)

        # convolution module
        residual = x
        x = self.norm_conv(x)

        x = residual + self.dropout(self.conv_module(x, key_padding_mask=key_padding_mask))

        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        return x


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class MaskedLmConformerEncoder(Module):
    r"""MaskedLmConformerEncoder is a stack of N encoder layers, modified from
        torch.nn.TransformerEncoder

    Args:
        encoder_layer: an instance of the MaskedLmConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = MaskedLmConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = MaskedLmConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> src, pos_emb = self.encoder_pos(src)
        >>> out = conformer_encoder(src, pos_emb)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer:  nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None):
        super(MaskedLmConformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args
            x: input of shape (T, N, C), i.e. (seq_len, batch,  channels)
            pos_emb: positional embedding tensor of shape (N, 2*T-1, C),
            attn_mask (optional, likely not used): mask for self-attention of
                  x to itself, of shape (T, T)
            key_padding_mask (optional): mask of shape (N, T), dtype must be bool.
        Returns:
            Returns a tensor with the same shape as x, i.e. (T, N, C).
        """
        for mod in self.layers:
            x = mod(
                x
                pos_emb,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        if self.norm is not None:
            x = self.norm(x)

        return x


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
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
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
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (1, 2*time-1, `*`).

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2
            + x.size(1),
        ]
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
            need_weights: if true, return (output, attn_output_weights); else, (output, None).

            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(T, N, C)` where T is the output sequence length, N is the batch size, C is
            the embedding dimension (number of channels).
            - key: :math:`(S, N, C)`, where S is the input sequence length.
            - value: :math:`(S, N, C)`
            - pos_emb: :math:`(N, 2*T-1, C)`. Note: this assumes T == S, which it will be, but
            still we use different letters because S relates to the input position,  T to the
            output posision.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the input sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(T, S)` where T is the output sequence length, S is the input sequence length.
            3D mask :math:`(N*num_heads, T, S)` where N is the batch size, where T is the output sequence length,
            S is the input sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

          Return:
            (output, attn_output_weights) if need_weights==True, else (output, None), where:

            - output: :math:`(T, N, C)` where T is the output sequence length, N is the batch size,
                C is the embedding/channel dimension.
            - attn_output_weights: :math:`(N, T, S)` where N is the batch size,
               T is the output sequence length, S is the input sequence length.
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
        )

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
            q, k, v = nn.functional.linear(
                query, in_proj_weight, in_proj_bias
            ).chunk(3, dim=-1)

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

        #if not self.is_espnet_structure:
        #    q = q * scaling

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
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct."
                    )
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
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

        #if not self.is_espnet_structure:
        #    attn_output_weights = (
        #        matrix_ac + matrix_bd
        #    )  # (batch, head, time1, time2)
        #else:

        attn_output_weights = (
            matrix_ac + matrix_bd
        ) * scaling  # (batch, head, time1, time2)

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
            .view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(
            attn_output, out_proj_weight, out_proj_bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None
