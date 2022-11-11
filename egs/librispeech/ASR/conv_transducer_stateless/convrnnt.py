#!/usr/bin/env python3

from typing import List, Optional, Tuple

import torch
from encoder_interface import EncoderInterface
from torch import nn

from icefall.utils import make_pad_mask


class ConvRNNT(EncoderInterface):
    """
    Args:
        num_features (int): Number of input features
        d_model (int): the output dimension
        num_global_cnn_encoder_layers (int): number of GlobalCNN Encoder
          layers.
        dropout (float): dropout rate
        layer_dropout (float): layer-dropout rate.
        aux_layer_period (int):
          Period of auxiliary layers used for random combiner during training.
          If set to 0, will not use the random combiner (Default).
          You can set a positive integer to use the random combiner, e.g., 3.
        causal (bool): Whether to use causal convolution in ConvRNNT encoder
          layer.
    """

    def __init__(
        self,
        num_features: int = 80,
        d_model: int = 512,
        num_global_cnn_encoder_layers: int = 6,
        num_lstm_encoder_layers: int = 7,
        lstm_hidden_size: int = 640,
        lstm_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        aux_layer_period: int = 0,
        causal: bool = True,
    ) -> None:
        super(ConvRNNT, self).__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.num_lstm_encoder_layers = num_lstm_encoder_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.causal = causal
        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T, d_model).
        # That is, it does only one thing:
        #   (1) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.local_cnn_encoder = LocalCNNEncoder(
            causal=causal,
        )

        self.global_cnn_encoder = GlobalCNNEncoder(
            encoder_layer=GlobalCNNEncoderLayer,
            channels=d_model,
            dropout=dropout,
            num_layers=num_global_cnn_encoder_layers,
            causal=causal,
        )

        self.transform = nn.Linear(2 * d_model, d_model)

        self.lstm_encoder = LstmEncoder(
            d_model=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_encoder_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
            dim_feedforward=lstm_dim_feedforward,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states:
            A tuple of 2 tensors (optional). It is for streaming inference.
            states[0] is the hidden states of all layers,
              with shape of (num_layers, N, lstm_hidden_size);
            states[1] is the cell states of all layers,
              with shape of (num_layers, N, lstm_hidden_size).
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          A tuple of 3 tensors:
            - embeddings: its shape is (N, T', d_model), where T' is the output
              sequence lengths.
            - lengths: a tensor of shape (batch_size,) containing the number of
              frames in `embeddings` before padding.
            - updated states, whose shape is the same as the input states.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        lengths = (((x_lens - 1) >> 1) - 1) >> 1

        if not torch.jit.is_tracing():
            assert x.size(0) == lengths.max().item()

        src_key_padding_mask = make_pad_mask(lengths)

        src_x = self.local_cnn_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (T, N, C)

        x = self.global_cnn_encoder(
            src_x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (T, N, C)

        x = self.transform(torch.cat((src_x, x), dim=2))

        if states is None:
            x = self.lstm_encoder(x, warmup=warmup)[0]
            new_states = (torch.empty(0), torch.empty(0))
        else:
            assert not self.training
            assert len(states) == 2
            if not torch.jit.is_tracing():
                # for hidden state
                assert states[0].shape == (
                    self.num_lstm_encoder_layers,
                    x.size(1),
                    self.lstm_hidden_size,
                )
                # for cell state
                assert states[1].shape == (
                    self.num_lstm_encoder_layers,
                    x.size(1),
                    self.lstm_hidden_size,
                )
            x, new_states = self.lstm_encoder(x, states)

        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return x, lengths, new_states

    @torch.jit.export
    def get_init_states(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get model initial states."""
        hidden_states = torch.zeros(
            (
                self.num_lstm_encoder_layers,
                batch_size,
                self.lstm_hidden_size,
            ),
            device=device,
        )
        cell_states = torch.zeros(
            (
                self.num_lstm_encoder_layers,
                batch_size,
                self.lstm_hidden_size,
            ),
            device=device,
        )
        return (hidden_states, cell_states)


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate=0.1):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        )

    def forward(self, x):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x


class GlobalCNNEncoder(nn.Module):
    r"""GlobalCNNEncoder is a stack of N GlobalCNNEncoder layers

    Args:
        encoder_layer: instance of GlobalCNNEncoderLayer() class (required)
        num_layers: the number of sub-encoder-layers
         in the GlobalCNNEncoder (required).

    Examples::
        >>> encoder_layer = GlobalCNNEncoderLayer(d_model=512)
        >>> global_cnn_encoder = GlobalCNNEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = global_cnn_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        channels: int = 512,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_layers: int = 6,
        aux_layers: Optional[List[int]] = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                encoder_layer(
                    channels=channels,
                    block_number=i + 1,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=causal,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

        self.aux_layers: List[int] = []
        self.combiner: Optional[nn.Module] = None
        if aux_layers is not None:
            assert len(set(aux_layers)) == len(aux_layers)
            assert num_layers - 1 not in aux_layers
            self.aux_layers = aux_layers + [num_layers - 1]
            self.combiner = RandomCombine(
                num_inputs=len(self.aux_layers),
                final_weight=0.5,
                pure_prob=0.333,
                stddev=2.0,
            )

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for src keys per batch (optional)

        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length,
            T is the target sequence length,
            N is the batch size,
            E is the feature number

        """
        output = src

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                src_key_padding_mask=src_key_padding_mask,
            )

        return output


class LocalCNNEncoder(nn.Module):
    """LocalCNNEncoder Module in ConvRNNT model.

    Args:
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).
        causal (bool): Whether to use causal convolution.

    """

    def __init__(
        self,
        kernel_size: int = 5,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Construct an LocalCNNEncoder object."""
        super(LocalCNNEncoder, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.causal = causal
        self.lorder = kernel_size - 1
        padding = (kernel_size - 1) // 2
        if self.causal:
            padding = 0

        self.conv1 = nn.Conv2d(
            1,
            100,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.conv2 = nn.Conv2d(
            100,
            100,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.conv3 = nn.Conv2d(
            100,
            64,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.conv4 = nn.Conv2d(
            64,
            64,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.activation = nn.ReLU()

        self.conv5 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for src keys per batch (optional)

        Returns:
            torch.Tensor: Output tensor (#time, batch, channels).
        """
        x = x.permute(1, 2, 0)  # (#batch, channels, time).
        if src_key_padding_mask is not None:
            x = x.masked_fill(
                src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0
            )
        x = x.unsqueeze(1)

        # Conv1
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv1(x)
        x = self.activation(x)

        # Conv2
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv2(x)
        x = self.activation(x)

        # Conv3
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv3(x)
        x = self.activation(x)

        # Conv4
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv4(x)
        x = self.activation(x)
        # Conv5
        # (N, 64, C, T) -> (N, 1, C, T) -> (N, C, T)
        x = self.conv5(x)
        x = x.squeeze(1).permute(2, 0, 1)
        return x


class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions
    Modified from rwightman`s squeeze_excite.py
    """

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        bias=True,
    ):
        super(SEModule, self).__init__()
        if not rd_channels:
            rd_channels = self.make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.fc1 = nn.Conv1d(channels, rd_channels, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x, src_key_padding_mask):
        x_se_num = src_key_padding_mask.eq(False).sum(axis=1, keepdim=True)
        x_se_num = x_se_num.unsqueeze(1)
        x_se = x.sum(axis=2, keepdim=True) / x_se_num
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

    def make_divisible(self, v, divisor=8, min_value=None, round_limit=0.9):
        min_value = min_value or divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < round_limit * v:
            new_v += divisor
        return new_v


class GlobalCNNEncoderLayer(nn.Module):
    """GlobalCNNEncoder Layer in ConvRNNT model.

    Args:
        channels (int): The number of channels of conv layers.
        block_number (int): The block number of stacked 6 blocks.
        kernel_size (int): Kernerl size of conv layers.
        dropout (float): dropout rate.
        layer_dropout (float): layer-dropout rate.
        bias (bool): Whether to use bias in conv layers (default=True).
        causal (bool): Whether to use causal convolution.
    """

    def __init__(
        self,
        channels: int,
        block_number: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Construct an GlobalCNNEncoder object."""
        super().__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.causal = causal
        self.layer_dropout = layer_dropout
        self.block_number = block_number

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.lorder = kernel_size - 1
        padding = (kernel_size - 1) // 2
        if self.causal:
            padding = 0

        self.depthwise_conv = nn.Conv1d(
            2 * channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
            dilation=2 ** block_number
        )

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.SE = SEModule(channels=channels)
        self.activation = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)
        self.out_norm1 = nn.BatchNorm1d(2 * channels)
        self.out_norm2 = nn.BatchNorm1d(channels)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for src keys per batch (optional)

        Returns:
            torch.Tensor: Output tensor (#time, batch, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time)
        out = x

        # GLU mechanism
        out = self.pointwise_conv1(out)  # (batch, 2*channels, time)
        out = self.activation(out)
        out = self.out_norm1(out)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            out = out.masked_fill(
                src_key_padding_mask.unsqueeze(1).expand_as(out), 0.0
            )
        if self.causal and self.lorder > 0:
            # Make depthwise_conv causal by
            # manualy padding self.lorder zeros to the left
            out = nn.functional.pad(
                out,
                ((2 ** self.block_number - 1) * 2 + self.lorder, 0),
                "constant",
                0.0,
            )
        out = self.depthwise_conv(out)
        out = self.activation(out)
        out = self.out_norm2(out)

        out = self.pointwise_conv2(out)  # (batch, channel, time)
        out = self.SE(out, src_key_padding_mask)
        out = self.Dropout(out) + x

        return out.permute(2, 0, 1)  # (N, C, T) -> (T, N, C)


def unstack_states(
    states: Tuple[torch.Tensor, torch.Tensor]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Unstack the lstm states corresponding to a batch of utterances into a list
    of states, where the i-th entry is the state from the i-th utterance.

    Args:
      states:
        A tuple of 2 elements.
        ``states[0]`` is the lstm hidden states, of a batch of utterance.
        ``states[1]`` is the lstm cell states, of a batch of utterances.

    Returns:
      A list of states.
        ``states[i]`` is a tuple of 2 elememts of i-th utterance.
        ``states[i][0]`` is the lstm hidden states of i-th utterance.
        ``states[i][1]`` is the lstm cell states of i-th utterance.
    """
    hidden_states, cell_states = states

    list_hidden_states = hidden_states.unbind(dim=1)
    list_cell_states = cell_states.unbind(dim=1)

    ans = [
        (h.unsqueeze(1), c.unsqueeze(1))
        for (h, c) in zip(list_hidden_states, list_cell_states)
    ]
    return ans


def stack_states(
    states_list: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack list of lstm states corresponding to separate utterances into a single
    lstm state so that it can be used as an input for lstm when those utterances
    are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponds to the lstm state for a single
        utterance.
        ``states[i]`` is a tuple of 2 elememts of i-th utterance.
        ``states[i][0]`` is the lstm hidden states of i-th utterance.
        ``states[i][1]`` is the lstm cell states of i-th utterance.


    Returns:
      A new state corresponding to a batch of utterances.
      It is a tuple of 2 elements.
        ``states[0]`` is the lstm hidden states, of a batch of utterance.
        ``states[1]`` is the lstm cell states, of a batch of utterances.
    """
    hidden_states = torch.cat([s[0] for s in states_list], dim=1)
    cell_states = torch.cat([s[1] for s in states_list], dim=1)
    ans = (hidden_states, cell_states)
    return ans


class LstmEncoder(nn.Module):
    """
    LstmEncoder is made up of lstm and feedforward networks.

    Args:
      d_model:
        The number of expected features in the input.(default=512)
      hidden_size:
        The hidden dimension of lstm layer.(default=640)
      num_layers:
        The number of lstm layers in the lstm encoder.(default=7)
      dropout:
        The dropout value (default=0.1)
      layer_dropout:
        The dropout value for model-level warmup (default=0.075)
      dim_feedforward:
        The dimension of feedforward network model (default=1024)
    """
    def __init__(
        self,
        d_model: int = 512,
        hidden_size: int = 640,
        num_layers: int = 7,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        bidirectional: bool = False,
        dim_feedforward: int = 1024,
    ) -> None:
        super().__init__()
        self.layer_dropout = layer_dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        assert hidden_size >= d_model, (hidden_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=d_model,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=False,
                    dropout=0.0,
                    bidirectional=bidirectional,
                )
                for i in range(num_layers)
            ]
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Pass the input through the lstm encoder layer

        Args:
          x:
            The sequence to the lstm encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          states:
            A tuple of 2 tensors (optional). It is for streaming inference.
            states[0] is the hidden states of all layers,
              with shape of (1, N, lstm_hidden_size);
            states[1] is the cell states of all layers,
              with shape of (1, N, lstm_hidden_size).
          warmup:
            It controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.
        """
        for i, mod in enumerate(self.layers):
            src = x

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

            if states is None:
                x = mod(x)[0]
                new_states = (torch.empty(0), torch.empty(0))
            else:
                assert not self.training
                assert len(states) == 2
                if not torch.jit.is_tracing():
                    # for hidden state
                    assert states[0].shape == (1, x.size(1), self.hidden_size)
                    # for cell state
                    assert states[1].shape == (1, x.size(1), self.hidden_size)
                x, new_states = mod(x, states)

            x = self.feed_forward(x)

            if alpha != 1.0:
                x = alpha * x + (1 - alpha) * src

        return x, new_states


class RandomCombine(nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.

    The idea is that the list of Tensors will be a list of outputs of multiple
    ConvRNNT layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        num_inputs: int,
        final_weight: float = 0.5,
        pure_prob: float = 0.5,
        stddev: float = 2.0,
    ) -> None:
        """
        Args:
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          final_weight:
            The amount of weight or probability we assign to the
            final layer when randomly choosing layers or when choosing
            continuous layer weights.
          pure_prob:
            The probability, on each frame, with which we choose
            only a single layer to output (rather than an interpolation)
          stddev:
            A standard deviation that we add to log-probs for computing
            randomized weights.

        The method of choosing which layers, or combinations of layers, to use,
        is conceptually as follows::

            With probability `pure_prob`::
               With probability `final_weight`: choose final layer,
               Else: choose random non-final layer.
            Else::
               Choose initial log-weights that correspond to assigning
               weight `final_weight` to the final layer and equal
               weights to other layers; then add Gaussian noise
               with variance `stddev` to these log-weights, and normalize
               to weights (note: the average weight assigned to the
               final layer here will not be `final_weight` if stddev>0).
        """
        super().__init__()
        assert 0 <= pure_prob <= 1, pure_prob
        assert 0 < final_weight < 1, final_weight
        assert num_inputs >= 1

        self.num_inputs = num_inputs
        self.final_weight = final_weight
        self.pure_prob = pure_prob
        self.stddev = stddev

        self.final_log_weight = (
            torch.tensor(
                (final_weight / (1 - final_weight)) * (self.num_inputs - 1)
            )
            .log()
            .item()
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels).  In test mode
          this is just the final input.
        """
        num_inputs = self.num_inputs
        assert len(inputs) == num_inputs
        if not self.training or torch.jit.is_scripting():
            return inputs[-1]

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        # weights: (num_frames, num_inputs)
        weights = self._get_random_weights(
            inputs[0].dtype, inputs[0].device, num_frames
        )

        weights = weights.reshape(num_frames, num_inputs, 1)
        # ans: (num_frames, num_channels, 1)
        ans = torch.matmul(stacked_inputs, weights)
        # ans: (*, num_channels)

        ans = ans.reshape(inputs[0].shape[:-1] + (num_channels,))

        # The following if causes errors for torch script in torch 1.6.0
        #  if __name__ == "__main__":
        #      # for testing only...
        #      print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans

    def _get_random_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ) -> torch.Tensor:
        """Return a tensor of random weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired
        Returns:
          A tensor of shape (num_frames, self.num_inputs), such that
          `ans.sum(dim=1)` is all ones.
        """
        pure_prob = self.pure_prob
        if pure_prob == 0.0:
            return self._get_random_mixed_weights(dtype, device, num_frames)
        elif pure_prob == 1.0:
            return self._get_random_pure_weights(dtype, device, num_frames)
        else:
            p = self._get_random_pure_weights(dtype, device, num_frames)
            m = self._get_random_mixed_weights(dtype, device, num_frames)
            return torch.where(
                torch.rand(num_frames, 1, device=device) < self.pure_prob, p, m
            )

    def _get_random_pure_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ):
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A one-hot tensor of shape `(num_frames, self.num_inputs)`, with
          exactly one weight equal to 1.0 on each frame.
        """
        final_prob = self.final_weight

        # final contains self.num_inputs - 1 in all elements
        final = torch.full((num_frames,), self.num_inputs - 1, device=device)
        # nonfinal contains random integers in [0..num_inputs - 2],
        # these are for non-final weights.
        nonfinal = torch.randint(
            self.num_inputs - 1, (num_frames,), device=device
        )

        indexes = torch.where(
            torch.rand(num_frames, device=device) < final_prob, final, nonfinal
        )
        ans = torch.nn.functional.one_hot(
            indexes, num_classes=self.num_inputs
        ).to(dtype=dtype)
        return ans

    def _get_random_mixed_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ):
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A tensor of shape (num_frames, self.num_inputs), which elements
          in [0..1] that sum to one over the second axis, i.e.
          `ans.sum(dim=1)` is all ones.
        """
        logprobs = (
            torch.randn(num_frames, self.num_inputs, dtype=dtype, device=device)
            * self.stddev
        )
        logprobs[:, -1] += self.final_log_weight
        return logprobs.softmax(dim=1)


def _test_random_combine(final_weight: float, pure_prob: float, stddev: float):
    print(
        f"_test_random_combine: final_weight={final_weight}, \
            pure_prob={pure_prob}, stddev={stddev}"
    )
    num_inputs = 3
    num_channels = 50
    m = RandomCombine(
        num_inputs=num_inputs,
        final_weight=final_weight,
        pure_prob=pure_prob,
        stddev=stddev,
    )

    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_random_combine_main():
    _test_random_combine(0.999, 0, 0.0)
    _test_random_combine(0.5, 0, 0.0)
    _test_random_combine(0.999, 0, 0.0)
    _test_random_combine(0.5, 0, 0.3)
    _test_random_combine(0.5, 1, 0.3)
    _test_random_combine(0.5, 0.5, 0.3)

    feature_dim = 80
    c = ConvRNNT(
        num_features=feature_dim,
        d_model=512,
        num_global_cnn_encoder_layers=6,
        lstm_hidden_size=640,
        lstm_dim_feedforward=1024,
        num_lstm_encoder_layers=7,
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        warmup=0.5,
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    feature_dim = 80
    c = ConvRNNT(
        num_features=feature_dim,
        d_model=512,
        num_global_cnn_encoder_layers=6,
        lstm_hidden_size=640,
        lstm_dim_feedforward=1024,
        num_lstm_encoder_layers=7,
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        warmup=0.5,
    )
    num_param = sum([p.numel() for p in c.parameters()])
    print(f"Number of model parameters: {num_param}")

    _test_random_combine_main()
