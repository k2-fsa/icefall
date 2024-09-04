from typing import Tuple


def get_padding(kernel_size, dilation=1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )
