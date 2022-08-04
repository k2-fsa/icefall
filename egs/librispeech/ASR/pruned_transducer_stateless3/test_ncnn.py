#!/usr/bin/env python3

from pathlib import Path

import ncnn
import numpy as np
import torch
from scaling import ScaledConv2d
from scaling_converter import scaled_conv2d_to_conv2d


def generate_scaled_conv2d():
    f = ScaledConv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    f = scaled_conv2d_to_conv2d(f)
    print(f)
    x = torch.rand(1, 1, 6, 8)  # NCHW
    m = torch.jit.trace(f, x)
    m.save("foo/scaled_conv2d.pt")
    print(m.graph)


def compare_scaled_conv2d():
    param = "foo/scaled_conv2d.ncnn.param"
    model = "foo/scaled_conv2d.ncnn.bin"

    with ncnn.Net() as net:
        with net.create_extractor() as ex:
            net = ncnn.Net()
            net.load_param(param)
            net.load_model(model)

            ex = net.create_extractor()
            x = torch.rand(1, 6, 5)  # CHW
            ex.input("in0", ncnn.Mat(x.numpy()).clone())
            ret, out0 = ex.extract("out0")
            assert ret == 0
            out0 = np.array(out0)
            out0 = torch.from_numpy(out0)

            m = torch.jit.load("foo/scaled_conv2d.pt")
            y = m(x.unsqueeze(0)).squeeze(0)

            assert torch.allclose(out0, y, atol=1e-3), (out0 - y).abs().max()


@torch.no_grad()
def main():
    if not Path("foo/scaled_conv2d.ncnn.param").is_file():
        generate_scaled_conv2d()
    else:
        compare_scaled_conv2d()


if __name__ == "__main__":
    torch.manual_seed(20220803)
    main()
