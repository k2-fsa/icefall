#!/usr/bin/env python3

import ncnn
import numpy as np
import torch


@torch.no_grad()
def main():
    x = torch.rand(30, 80, dtype=torch.float32)  # (T, C)
    x_lens = torch.tensor([30])
    m = torch.jit.load("foo/conformer.pt")
    t, t_lens, t_embed = m(x.unsqueeze(0), x_lens)
    t = t.squeeze(1)
    t_embed = t_embed.squeeze(0)

    print(t.shape)
    print(t_lens)
    print(t_embed.shape)

    param = "foo/conformer.ncnn.param"
    model = "foo/conformer.ncnn.bin"
    with ncnn.Net() as net:
        net.load_param(param)
        net.load_model(model)
        with net.create_extractor() as ex:
            x = x.to(torch.float32)

            # ncnn only support float binary ops
            x_lens = x_lens.to(torch.float32)

            ex.input("in0", ncnn.Mat(x.numpy()).clone())
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            n = np.array(ncnn_out0)
            print(n.shape)
            n = torch.from_numpy(n)
            print(n.reshape(-1)[:10])
            print(t.reshape(-1)[:10])
            assert torch.allclose(t, n, atol=1e-3), (t - n).abs().max()

            # test length
            ret, ncnn_out1 = ex.extract("out1")
            assert ret == 0, ret
            n_lens = np.array(ncnn_out1)
            assert t_lens.item() == n_lens.item()

            # test pos_emb
            ret, ncnn_out2 = ex.extract("out2")
            assert ret == 0, ret
            n_embed = np.array(ncnn_out2)
            n_embed = torch.from_numpy(n_embed)
            print(n_embed.reshape(-1)[:10])
            print(t_embed.reshape(-1)[:10])
            assert torch.allclose(t_embed, n_embed, atol=1e-3), (
                (t - n).abs().max()
            )
            assert torch.allclose(t, n, atol=1e-3), (t - n).abs().max()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if __name__ == "__main__":
    torch.manual_seed(202208010)
    main()
