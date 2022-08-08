#!/usr/bin/env python3

import math

import ncnn
import numpy as np
import torch

LOG_EPS = math.log(1e-10)


@torch.no_grad()
def main():
    x = torch.rand(10, 3)
    f = torch.jit.load("foo/encoder_pos.pt")

    param = "foo/encoder_pos.ncnn.param"
    model = "foo/encoder_pos.ncnn.bin"

    with ncnn.Net() as net:
        net.load_param(param)
        net.load_model(model)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy()).clone())
            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            ncnn_out0 = np.array(ncnn_out0)

            ret, ncnn_out1 = ex.extract("out1")
            assert ret == 0, ret
            ncnn_out1 = np.array(ncnn_out1)

            torch_out0, torch_out1 = f(x.unsqueeze(0))
            torch_out0 = torch_out0.squeeze(0)
            torch_out1 = torch_out1.squeeze(1)

            ncnn_out0 = torch.from_numpy(ncnn_out0)
            ncnn_out1 = torch.from_numpy(ncnn_out1)

            torch.allclose(torch_out0, ncnn_out0), (
                torch_out0 - ncnn_out0
            ).abs().max()
            torch.allclose(torch_out1, ncnn_out1), (
                torch_out1 - ncnn_out1
            ).abs().max()


if __name__ == "__main__":
    main()
