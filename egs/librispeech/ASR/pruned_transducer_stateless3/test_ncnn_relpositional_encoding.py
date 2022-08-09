#!/usr/bin/env python3


import ncnn
import numpy as np
import torch


@torch.no_grad()
def main():
    x = torch.rand(100, 512)  # (T, C)
    m = torch.jit.load("foo/encoder_pos.pt")
    _, t = m(x.unsqueeze(0))  # bach size is 1
    t = t.squeeze(0)  # (T, C)

    param = "foo/encoder_pos.ncnn.param"
    model = "foo/encoder_pos.ncnn.bin"
    with ncnn.Net() as net:
        net.load_param(param)
        net.load_model(model)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out1")
            assert ret == 0, ret
            n = np.array(ncnn_out0)
            print(n.shape)  # (6, 512), (T, C)
            n = torch.from_numpy(n)

            print(t.reshape(-1)[:10])
            print(n.reshape(-1)[:10])
            assert torch.allclose(t, n), (t - n).abs().max()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if __name__ == "__main__":
    torch.manual_seed(20220808)
    main()
