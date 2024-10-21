#!/usr/bin/env python3

import json
import logging

import torch
from inference import get_parser
from tokenizer import Tokenizer
from train import get_model, get_params
from icefall.checkpoint import load_checkpoint
from onnxruntime.quantization import QuantType, quantize_dynamic


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        temperature: torch.Tensor,
        length_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args: :
          x: (batch_size, num_tokens), torch.int64
          x_lengths: (batch_size,), torch.int64
          temperature: (1,), torch.float32
          length_scale (1,), torch.float32
        Returns:
          mel: (batch_size, feat_dim, num_frames)

        """
        mel = self.model.synthesise(
            x=x,
            x_lengths=x_lengths,
            n_timesteps=3,
            temperature=temperature,
            length_scale=length_scale,
        )["mel"]

        # mel: (batch_size, feat_dim, num_frames)

        return mel


@torch.inference_mode
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()

    params.update(vars(args))

    tokenizer = Tokenizer(params.tokens)
    params.blank_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size
    params.model_args.n_vocab = params.vocab_size

    with open(params.cmvn) as f:
        stats = json.load(f)
        params.data_args.data_statistics.mel_mean = stats["fbank_mean"]
        params.data_args.data_statistics.mel_std = stats["fbank_std"]

        params.model_args.data_statistics.mel_mean = stats["fbank_mean"]
        params.model_args.data_statistics.mel_std = stats["fbank_std"]
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    wrapper = ModelWrapper(model)
    wrapper.eval()

    # Use a large value so the the rotary position embedding in the text
    # encoder has a large initial length
    x = torch.ones(1, 2000, dtype=torch.int64)
    x_lengths = torch.tensor([x.shape[1]], dtype=torch.int64)
    temperature = torch.tensor([1.0])
    length_scale = torch.tensor([1.0])
    mel = wrapper(x, x_lengths, temperature, length_scale)
    print("mel", mel.shape)

    opset_version = 14
    filename = "model.onnx"
    torch.onnx.export(
        wrapper,
        (x, x_lengths, temperature, length_scale),
        filename,
        opset_version=opset_version,
        input_names=["x", "x_length", "temperature", "length_scale"],
        output_names=["mel"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_length": {0: "N"},
            "mel": {0: "N", 2: "L"},
        },
    )

    print("Generate int8 quantization models")

    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QInt8,
    )

    print(f"Saved to {filename} and {filename_int8}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
