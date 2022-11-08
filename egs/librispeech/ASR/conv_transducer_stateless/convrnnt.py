#!/usr/bin/env python3

from typing import List, Optional, Tuple

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)
from torch import Tensor, nn

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
        causal (bool): Whether to use causal convolution in ConvRNNT encoder
            layer.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 512,
        local_cnn_encoder_kernel_size: int = 5,
        num_global_cnn_encoder_layers: int = 6,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        aux_layer_period: int = 3,
        causal: bool = True,
    ) -> None:
        super(ConvRNNT, self).__init__()

        self.num_features = num_features

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T, d_model).
        # That is, it does only one thing:
        #   (1) embedding: num_features -> d_model
        self.encoder_embed = Embedding(num_features, d_model)
        self.encoder_layers = num_global_cnn_encoder_layers
        self.d_model = d_model
        self.causal = causal

        self.local_cnn_encoder = LocalCNNEncoder(
            kernel_size=local_cnn_encoder_kernel_size,
            causal=causal,
        )

        # aux_layers from 1/3
        self.global_cnn_encoder = GlobalCNNEncoder(
            encoder_layer=GlobalCNNEncoderLayer,
            channels=d_model,
            dropout=dropout,
            num_layers=num_global_cnn_encoder_layers,
            aux_layers=list(
                range(
                    num_global_cnn_encoder_layers // 3,
                    num_global_cnn_encoder_layers - 1,
                    aux_layer_period,
                )
            ),
            causal=causal,
        )
        self._init_state: List[torch.Tensor] = [torch.empty(0)]

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, d_model)
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        lengths = x_lens
        assert x.size(0) == lengths.max().item()
        src_key_padding_mask = make_pad_mask(lengths)

        src_x = self.local_cnn_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (T, N, C)

        x = self.global_cnn_encoder(
            src_x,
            src_key_padding_mask=src_key_padding_mask,
            warmup=warmup,
        )  # (T, N, C)

        x = torch.cat((src_x, x), dim=2)
        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return x, lengths

    @torch.jit.export
    def get_init_state(
        self, device: torch.device
    ) -> List[torch.Tensor]:
        """Return the initial cache state of the model.
        Returns:
          Return the initial state of the model, it is a list containing one
          tensors, which is the cache of conv_modules which has a shape of
          (num_global_cnn_encoder_layers, encoder_dim).
          NOTE: the returned tensors are on the given device.
        """
        if (
            len(self._init_state) == 2
        ):
            # Note: It is OK to share the init state as it is
            # not going to be modified by the model
            return self._init_state

        init_states: List[torch.Tensor] = [
            torch.zeros(
                (
                    self.encoder_layers,
                    self.d_model,
                ),
                device=device,
            ),
        ]

        self._init_state = init_states

        return init_states


class GlobalCNNEncoder(nn.Module):
    r"""GlobalCNNEncoder is a stack of N GlobalCNNEncoder layers

    Args:
        encoder_layer: instance of GlobalCNNEncoderLayer() class (required)
        num_layers: the number of sub-encoder-layers
         in the GlobalCNNEncoder (required).
        warmup: controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.

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
        aux_layers: List[int] = None,
        causal: bool = True,
        warmup: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                encoder_layer(
                    channels=channels,
                    block_number=i+1,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=causal,
                    warmup=warmup,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

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
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        warmup: float = 1.0,
    ) -> Tensor:
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

        outputs = []

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                src_key_padding_mask=src_key_padding_mask,
            )
            if i in self.aux_layers:
                outputs.append(output)

        output = self.combiner(outputs)

        return output


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Modified from rwightman`s squeeze_excite.py
    """
    def __init__(
        self,
        channels,
        rd_ratio=1. / 16,
        rd_channels=None,
        rd_divisor=8,
        add_maxpool=False,
        bias=True,
        act_layer=nn.ReLU,
        norm_layer=None,
        gate_layer='Sigmoid',
    ):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = self.make_divisible(
                channels * rd_ratio,
                rd_divisor,
                round_limit=0.
            )
        self.fc1 = nn.Conv1d(channels, rd_channels, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x, src_key_padding_mask):
        x_se_num = src_key_padding_mask.eq(False).sum(axis=1, keepdim=True)
        x_se_num = x_se_num.unsqueeze(1)
        x_se = x.sum(axis=2, keepdim=True) / x_se_num
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax(2, keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

    def make_divisible(self, v, divisor=8, min_value=None, round_limit=.9):
        min_value = min_value or divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < round_limit * v:
            new_v += divisor
        return new_v


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

        self.conv1 = ScaledConv2d(
            1,
            100,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.deriv_balancer1 = ActivationBalancer(
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )

        self.conv2 = ScaledConv2d(
            100,
            100,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.conv3 = ScaledConv2d(
            100,
            64,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.deriv_balancer3 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.conv4 = ScaledConv2d(
            64,
            64,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2, padding),
            bias=bias,
        )

        self.deriv_balancer4 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.activation = DoubleSwish()

        self.conv5 = ScaledConv2d(
            64,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for src keys per batch (optional)

        Returns:
            Tensor: Output tensor (#time, batch, channels).
        """
        x = x.permute(1, 2, 0)  # (#batch, channels, time).
        if src_key_padding_mask is not None:
            x = x.masked_fill(
                src_key_padding_mask.unsqueeze(1).expand_as(x),
                0.0
            )
        x = x.unsqueeze(1)

        # Conv1
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv1(x)
        x = self.deriv_balancer1(x)
        x = self.activation(x)

        # Conv2
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv2(x)
        x = self.deriv_balancer2(x)
        x = self.activation(x)

        # Conv3
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv3(x)
        x = self.deriv_balancer3(x)
        x = self.activation(x)

        # Conv4
        if self.causal and self.lorder > 0:
            # Make conv causal by
            # manualy padding self.lorder zeros to the left
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        x = self.conv4(x)
        x = self.deriv_balancer4(x)
        x = self.activation(x)
        # Conv5
        # (N, 64, C, T) -> (N, 1, C, T) -> (N, C, T)
        x = self.conv5(x)
        x = x.squeeze(1).permute(2, 0, 1)
        return x


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
        warmup: float = 1.0,
    ) -> None:
        """Construct an GlobalCNNEncoder object."""
        super().__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.causal = causal
        self.warmup = warmup
        self.layer_dropout = layer_dropout
        self.block_number = block_number

        self.pointwise_conv1 = ScaledConv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.deriv_balancer1 = ActivationBalancer(
            channel_dim=1,
            max_abs=10.0,
            min_positive=0.05,
            max_positive=1.0,
        )

        self.lorder = kernel_size - 1
        padding = (kernel_size - 1) // 2
        if self.causal:
            padding = 0

        self.depthwise_conv = ScaledConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
            dilation=2**block_number
        )

        self.deriv_balancer2 = ActivationBalancer(
            channel_dim=1,
            min_positive=0.05,
            max_positive=1.0,
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.25,
        )

        self.SE = SEModule(channels=channels)
        self.Dropout = nn.Dropout(dropout)
        self.BN = nn.BatchNorm1d(channels)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for src keys per batch (optional)
            warmup: controls selective bypass of of layers; if < 1.0, we will
              bypass layers more frequently.

        Returns:
            Tensor: Output tensor (#time, batch, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).
        out = x

        warmup_scale = min(0.1 + self.warmup, 1.0)
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

        # GLU mechanism
        out = self.pointwise_conv1(out)  # (batch, 2*channels, time)
        out = self.deriv_balancer1(out)
        out = nn.functional.glu(out, dim=1)  # (batch, channels, time)
        out = self.BN(out)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            out = out.masked_fill(
                src_key_padding_mask.unsqueeze(1).expand_as(out),
                0.0
            )
        if self.causal and self.lorder > 0:
            # Make depthwise_conv causal by
            # manualy padding self.lorder zeros to the left
            out = nn.functional.pad(
                out,
                ((2**self.block_number - 1) * 2 + self.lorder, 0),
                "constant",
                0.0
            )
        out = self.depthwise_conv(out)
        out = self.deriv_balancer2(out)
        out = self.activation(out)
        out = self.BN(out)

        out = self.pointwise_conv2(out)  # (batch, channel, time)
        out = self.SE(out, src_key_padding_mask)
        out = self.Dropout(out) + x

        if alpha != 1.0:
            out = alpha * out + (1 - alpha) * x

        return out.permute(2, 0, 1)  # (N, C, T) -> (T, N, C)


class Embedding(nn.Module):
    """Embedding

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T, odim)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Args:
          in_channels:
            Number of input features. The input shape is (N, T, in_channels).
          out_channels:
            Output dim. The output shape is (N, T, out_channels)
        """
        super().__init__()
        self.out = ScaledLinear(
            in_channels, out_channels
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        self.out_norm = BasicNorm(out_channels, learn_eps=False)
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, T, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, T, odim)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x


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
    ) -> Tensor:
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

    feature_dim = 50
    c = ConvRNNT(num_features=feature_dim, d_model=512)
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    feature_dim = 50
    c = ConvRNNT(num_features=feature_dim, d_model=512)
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        warmup=0.5,
    )

    _test_random_combine_main()