#!/usr/bin/env python3

import math

import ncnn
import numpy as np
import torch

LOG_EPS = math.log(1e-10)


@torch.no_grad()
def main():
    x = torch.rand(1, 200, 80)
    f = torch.jit.load("foo/encoder_embed.pt")

    param = "foo/encoder_embed.ncnn.param"
    model = "foo/encoder_embed.ncnn.bin"

    with ncnn.Net() as net:
        net.load_param(param)
        net.load_model(model)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy()).clone())
            ret, out0 = ex.extract("out0")
            assert ret == 0
            out0 = np.array(out0)
            print("ncnn", out0.shape)
            t = f(x)
            out0 = torch.from_numpy(out0)
            t = t.squeeze(0)
            print("torch", t.shape)
            torch.allclose(out0, t), (t - out0).abs().max()


if __name__ == "__main__":
    main()
