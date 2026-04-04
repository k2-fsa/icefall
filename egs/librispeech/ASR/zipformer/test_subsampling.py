#!/usr/bin/env python3

import torch
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling


def test_conv2d_subsampling():
    layer1_channels = 8
    layer2_channels = 32
    layer3_channels = 128

    out_channels = 192
    encoder_embed = Conv2dSubsampling(
        in_channels=80,
        out_channels=out_channels,
        layer1_channels=layer1_channels,
        layer2_channels=layer2_channels,
        layer3_channels=layer3_channels,
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    N = 2
    T = 200
    num_features = 80
    x = torch.rand(N, T, num_features)
    x_copy = x.clone()

    x = x.unsqueeze(1)  # (N, 1, T, num_features)

    x = encoder_embed.conv[0](x)  # conv2d, in 1, out 8, kernel 3, padding (0,1)
    assert x.shape == (N, layer1_channels, T - 2, num_features)
    # (2, 8, 198, 80)

    x = encoder_embed.conv[1](x)  # scale grad
    x = encoder_embed.conv[2](x)  # balancer
    x = encoder_embed.conv[3](x)  # swooshR

    x = encoder_embed.conv[4](x)  # conv2d, in 8, out 32, kernel 3, stride 2
    assert x.shape == (
        N,
        layer2_channels,
        ((T - 2) - 3) // 2 + 1,
        (num_features - 3) // 2 + 1,
    )
    # (2, 32, 98, 39)

    x = encoder_embed.conv[5](x)  # balancer
    x = encoder_embed.conv[6](x)  # swooshR

    # conv2d:
    # in 32, out 128, kernel 3, stride (1, 2)
    x = encoder_embed.conv[7](x)
    assert x.shape == (
        N,
        layer3_channels,
        (((T - 2) - 3) // 2 + 1) - 2,
        (((num_features - 3) // 2 + 1) - 3) // 2 + 1,
    )
    # (2, 128, 96, 19)

    x = encoder_embed.conv[8](x)  # balancer
    x = encoder_embed.conv[9](x)  # swooshR

    # (((T - 2) - 3) // 2 + 1) - 2
    # = (T - 2) - 3) // 2 + 1 - 2
    # = ((T - 2) - 3) // 2 - 1
    # = (T - 2 - 3) // 2 - 1
    # = (T - 5) // 2 - 1
    # = (T - 7) // 2
    assert x.shape[2] == (x_copy.shape[1] - 7) // 2

    # (((num_features - 3) // 2 + 1) - 3) // 2 + 1,
    # = ((num_features - 3) // 2 + 1 - 3) // 2 + 1,
    # = ((num_features - 3) // 2 - 2) // 2 + 1,
    # = (num_features - 3 - 4) // 2 // 2 + 1,
    # = (num_features - 7) // 2 // 2 + 1,
    # = (num_features - 7) // 4 + 1,
    # = (num_features - 3) // 4
    assert x.shape[3] == (x_copy.shape[2] - 3) // 4

    assert x.shape == (N, layer3_channels, (T - 7) // 2, (num_features - 3) // 4)

    # Input shape to convnext is
    #
    # (N, layer3_channels, (T-7)//2, (num_features - 3)//4)

    # conv2d: in layer3_channels, out layer3_channels, groups layer3_channels
    # kernel_size 7, padding 3
    x = encoder_embed.convnext.depthwise_conv(x)
    assert x.shape == (N, layer3_channels, (T - 7) // 2, (num_features - 3) // 4)

    # conv2d: in layer3_channels, out hidden_ratio * layer3_channels, kernel_size 1
    x = encoder_embed.convnext.pointwise_conv1(x)
    assert x.shape == (N, layer3_channels * 3, (T - 7) // 2, (num_features - 3) // 4)

    x = encoder_embed.convnext.hidden_balancer(x)  # balancer
    x = encoder_embed.convnext.activation(x)  # swooshL

    # conv2d: in hidden_ratio * layer3_channels, out layer3_channels, kernel 1
    x = encoder_embed.convnext.pointwise_conv2(x)
    assert x.shape == (N, layer3_channels, (T - 7) // 2, (num_features - 3) // 4)

    # bypass and layer drop, omitted here.
    x = encoder_embed.convnext.out_balancer(x)

    # Note: the input and output shape of ConvNeXt are the same

    x = x.transpose(1, 2).reshape(N, (T - 7) // 2, -1)
    assert x.shape == (N, (T - 7) // 2, layer3_channels * ((num_features - 3) // 4))

    x = encoder_embed.out(x)
    assert x.shape == (N, (T - 7) // 2, out_channels)

    x = encoder_embed.out_whiten(x)
    x = encoder_embed.out_norm(x)
    # final layer is dropout

    # test streaming forward

    subsampling_factor = 2
    cached_left_padding = encoder_embed.get_init_states(batch_size=N)
    depthwise_conv_kernel_size = 7
    pad_size = (depthwise_conv_kernel_size - 1) // 2

    assert cached_left_padding.shape == (
        N,
        layer3_channels,
        pad_size,
        (num_features - 3) // 4,
    )

    chunk_size = 16
    right_padding = pad_size * subsampling_factor
    T = chunk_size * subsampling_factor + 7 + right_padding
    x = torch.rand(N, T, num_features)
    x_lens = torch.tensor([T] * N)
    y, y_lens, next_cached_left_padding = encoder_embed.streaming_forward(
        x, x_lens, cached_left_padding
    )

    assert y.shape == (N, chunk_size, out_channels), y.shape
    assert next_cached_left_padding.shape == cached_left_padding.shape

    assert y.shape[1] == y_lens[0] == y_lens[1]


def main():
    test_conv2d_subsampling()


if __name__ == "__main__":
    main()
