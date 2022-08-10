#!/usr/bin/env python3

import ncnn
import torch


@torch.no_grad()
def main():
    x = torch.tensor([1, 3, 5, 8])
    m = torch.jit.load("foo/make_pad_mask.pt")
    t = m(x)
    print(t.shape)
    print(t)

    param = "foo/make_pad_mask.ncnn.param"
    model = "foo/make_pad_mask.ncnn.bin"
    with ncnn.Net() as net:
        net.load_param(param)
        net.load_model(model)
        with net.create_extractor() as ex:
            x = x.to(torch.int32)
            ex.input("in0", ncnn.Mat(x.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            n = ncnn_out0.numpy("i")
            print(n.shape)
            n = torch.from_numpy(n).to(torch.bool)
            print(n)
            assert torch.equal(t, n), (t, n)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if __name__ == "__main__":
    torch.manual_seed(202208010)
    main()
