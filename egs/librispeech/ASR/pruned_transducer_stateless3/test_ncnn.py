#!/usr/bin/env python3


import torch
import torch.nn as nn
from conformer import RelPositionalEncoding
from scaling_converter import convert_scaled_to_non_scaled
from train import get_params, get_transducer_model


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 100)
        self.encoder_pos = RelPositionalEncoding(100, 0.1)
        self.linear2 = nn.Linear(100, 2)

    def forward(self, x):
        y = self.linear(x)
        z, embed = self.encoder_pos(y)
        return z, embed


def test():
    f = Foo()
    f.eval()
    #  f.encoder_pos.for_ncnn = True
    x = torch.rand(1, 10, 3)
    y, _ = f(x)
    print(y.shape)
    #  print(embed.shape)

    m = torch.jit.trace(f, x)
    m.save("foo/encoder_pos.pt")
    print(m.graph)
    #  print(m.encoder_pos.graph)


def get_model():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2

    params.dynamic_chunk_training = False
    params.short_chunk_size = 25
    params.num_left_chunks = 4
    params.causal_convolution = False

    model = get_transducer_model(params, enable_giga=False)
    return model


def test_encoder_embedding():
    model = get_model()
    model.eval()
    model = convert_scaled_to_non_scaled(model)
    f = model.encoder
    f.for_ncnn = True
    f.encoder_pos.for_ncnn = True

    f.for_ncnn = True
    x = torch.rand(1, 100, 80)  # NTC
    x_lens = torch.tensor([100])
    m = torch.jit.trace(f, (x, x_lens))
    m.save("foo/encoder_pos.pt")
    print(m.graph)


@torch.no_grad()
def main():
    #  test_encoder_embedding()
    test()


if __name__ == "__main__":
    torch.manual_seed(20220803)
    main()
