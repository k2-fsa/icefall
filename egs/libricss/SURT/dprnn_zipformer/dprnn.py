import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from scaling import ActivationBalancer, BasicNorm, DoubleSwish, ScaledLinear, ScaledLSTM
from torch.autograd import Variable

EPS = torch.finfo(torch.get_default_dtype()).eps


def _pad_segment(input, segment_size):
    # Source: https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dprnn.py#L342
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, segment_size):
    # Source: https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dprnn.py#L358
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = _pad_segment(input, segment_size)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = (
        input[:, :, :-segment_stride]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments2 = (
        input[:, :, segment_stride:]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments = (
        torch.cat([segments1, segments2], 3)
        .view(batch_size, dim, -1, segment_size)
        .transpose(2, 3)
    )

    return segments.contiguous(), rest


def merge_feature(input, rest):
    # Source: https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dprnn.py#L385
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
    )  # B, N, K, L

    input1 = (
        input[:, :, :, :segment_size]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, segment_stride:]
    )
    input2 = (
        input[:, :, :, segment_size:]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, :-segment_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T


class RNNEncoderLayer(nn.Module):
    """
    RNNEncoderLayer is made up of lstm and feedforward networks.
    Args:
      input_size:
        The number of expected features in the input (required).
      hidden_size:
        The hidden dimension of rnn layer.
      dropout:
        The dropout value (default=0.1).
      layer_dropout:
        The dropout value for model-level warmup (default=0.075).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super(RNNEncoderLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        assert hidden_size >= input_size, (hidden_size, input_size)
        self.lstm = ScaledLSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2 if bidirectional else hidden_size,
            proj_size=0,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.norm_final = BasicNorm(input_size)

        # try to ensure the output is close to zero-mean (or at least, zero-median).  # noqa
        self.balancer = ActivationBalancer(
            num_channels=input_size,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pass the input through the encoder layer.
        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          states:
            A tuple of 2 tensors (optional). It is for streaming inference.
            states[0] is the hidden states of all layers,
              with shape of (1, N, input_size);
            states[1] is the cell states of all layers,
              with shape of (1, N, hidden_size).
        """
        src_orig = src

        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        alpha = warmup if self.training else 1.0

        # lstm module
        src_lstm, new_states = self.lstm(src, states)
        src = self.dropout(src_lstm) + src
        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src


# dual-path RNN
class DPRNN(nn.Module):
    """Deep dual-path RNN.
    Source: https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dprnn.py

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_blocks: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(
        self,
        feature_dim,
        input_size,
        hidden_size,
        output_size,
        dropout=0.1,
        num_blocks=1,
        segment_size=50,
        chunk_width_randomization=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.segment_size = segment_size
        self.chunk_width_randomization = chunk_width_randomization

        self.input_embed = nn.Sequential(
            ScaledLinear(feature_dim, input_size),
            BasicNorm(input_size),
            ActivationBalancer(
                num_channels=input_size,
                channel_dim=-1,
                min_positive=0.45,
                max_positive=0.55,
            ),
        )

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        for _ in range(num_blocks):
            # intra-RNN is non-causal
            self.row_rnn.append(
                RNNEncoderLayer(
                    input_size, hidden_size, dropout=dropout, bidirectional=True
                )
            )
            self.col_rnn.append(
                RNNEncoderLayer(
                    input_size, hidden_size, dropout=dropout, bidirectional=False
                )
            )

        # output layer
        self.out_embed = nn.Sequential(
            ScaledLinear(input_size, output_size),
            BasicNorm(output_size),
            ActivationBalancer(
                num_channels=output_size,
                channel_dim=-1,
                min_positive=0.45,
                max_positive=0.55,
            ),
        )

    def forward(self, input):
        # input shape: B, T, F
        input = self.input_embed(input)
        B, T, D = input.shape

        if self.chunk_width_randomization and self.training:
            segment_size = random.randint(self.segment_size // 2, self.segment_size)
        else:
            segment_size = self.segment_size
        input, rest = split_feature(input.transpose(1, 2), segment_size)
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            output = (
                output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            )  # B, N, dim1, dim2

            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            output = (
                output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            )  # B, N, dim1, dim2

        output = merge_feature(output, rest)
        output = output.transpose(1, 2)
        output = self.out_embed(output)

        # Apply ReLU to the output
        output = torch.relu(output)

        return output


if __name__ == "__main__":

    model = DPRNN(
        80,
        256,
        256,
        160,
        dropout=0.1,
        num_blocks=4,
        segment_size=32,
        chunk_width_randomization=True,
    )
    input = torch.randn(2, 1002, 80)
    print(sum(p.numel() for p in model.parameters()))
    print(model(input).shape)
