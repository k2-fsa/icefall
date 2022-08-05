#!/usr/bin/env python3


import torch
from scaling_converter import convert_scaled_to_non_scaled
from train import get_params, get_transducer_model


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
    model = convert_scaled_to_non_scaled(model)

    f = model.encoder.encoder_embed
    f.for_ncnn = True
    print(f)
    x = torch.rand(1, 100, 80)  # NTC
    m = torch.jit.trace(f, x)
    m.save("foo/encoder_embed.pt")
    print(m.graph)


@torch.no_grad()
def main():
    test_encoder_embedding()


if __name__ == "__main__":
    torch.manual_seed(20220803)
    main()
