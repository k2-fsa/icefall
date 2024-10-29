#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import logging
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
from inference import load_vocoder


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args: :
          mel: (batch_size, feat_dim, num_frames), torch.float32
        Returns:
          audio: (batch_size, num_samples), torch.float32
        """
        audio = self.model(mel).clamp(-1, 1).squeeze(1)
        return audio


@torch.inference_mode()
def main():
    # Please go to
    # https://github.com/csukuangfj/models/tree/master/hifigan
    # to download the following files
    model_filenames = ["./generator_v1", "./generator_v2", "./generator_v3"]

    for f in model_filenames:
        logging.info(f)
        if not Path(f).is_file():
            logging.info(f"Skipping {f} since {f} does not exist")
            continue
        model = load_vocoder(f)
        wrapper = ModelWrapper(model)
        wrapper.eval()
        num_param = sum([p.numel() for p in wrapper.parameters()])
        logging.info(f"{f}: Number of parameters: {num_param}")

        # Use a large value so the rotary position embedding in the text
        # encoder has a large initial length
        x = torch.ones(1, 80, 100000, dtype=torch.float32)
        opset_version = 14
        suffix = f.split("_")[-1]
        filename = f"hifigan_{suffix}.onnx"
        torch.onnx.export(
            wrapper,
            x,
            filename,
            opset_version=opset_version,
            input_names=["mel"],
            output_names=["audio"],
            dynamic_axes={
                "mel": {0: "N", 2: "L"},
                "audio": {0: "N", 1: "L"},
            },
        )

        meta_data = {
            "model_type": "hifigan",
            "model_filename": f.split("/")[-1],
            "sample_rate": 22050,
            "version": 1,
            "model_author": "jik876",
            "maintainer": "k2-fsa",
            "dataset": "LJ Speech",
            "url1": "https://github.com/jik876/hifi-gan",
            "url2": "https://github.com/csukuangfj/models/tree/master/hifigan",
        }
        add_meta_data(filename=filename, meta_data=meta_data)
        print(meta_data)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
