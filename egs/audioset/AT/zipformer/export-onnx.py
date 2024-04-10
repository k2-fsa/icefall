#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang, Wei Kang)
# Copyright 2023 Danqing Fu (danqing.fu@gmail.com)

"""
This script exports a transducer model from PyTorch to ONNX.

Usage of this script:

  repo_url=https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12
  repo=$(basename $repo_url)
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  pushd $repo/exp
  git lfs pull --include pretrained.pt
  ln -s pretrained.pt epoch-99.pt
  popd

  python3 zipformer/export-onnx.py \
      --exp-dir $repo/exp \
      --epoch 99 \
      --avg 1 \
      --use-averaged-model 0

  pushd $repo/exp
  mv model-epoch-99-avg-1.onnx model.onnx
  mv model-epoch-99-avg-1.int8.onnx model.int8.onnx
  popd

See ./onnx_pretrained.py
use the exported ONNX models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import k2
import onnx
import onnxoptimizer
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_model, get_params
from zipformer import Zipformer2

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import make_pad_mask, num_tokens, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for averaging.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    add_model_arguments(parser)

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


class OnnxAudioTagger(nn.Module):
    """A wrapper for Zipformer audio tagger"""

    def __init__(
        self, encoder: Zipformer2, encoder_embed: nn.Module, classifier: nn.Linear
    ):
        """
        Args:
          encoder:
            A Zipformer encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.classifier = classifier

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Please see the help information of Zipformer.forward

        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 1-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a tensor containing:
            - probs, A 2-D tensor of shape (N, num_classes)

        """
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)  # (N,T,C)

        logits = self.classifier(encoder_out)  # (N, T, num_classes)
        # Note that this is slightly different from model.py for better
        # support of onnx
        logits = logits.mean(dim=1)
        probs = logits.sigmoid()
        return probs


def export_audio_tagging_model_onnx(
    model: OnnxAudioTagger,
    filename: str,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T', joiner_dim)
        - encoder_out_lens, a tensor of shape (N,)

    Args:
      model:
        The input encoder model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    x = torch.zeros(1, 200, 80, dtype=torch.float32)
    x_lens = torch.tensor([200], dtype=torch.int64)

    model = torch.jit.trace(model, (x, x_lens))

    torch.onnx.export(
        model,
        (x, x_lens),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "probs": {0: "N"},
        },
    )

    meta_data = {
        "model_type": "zipformer2",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "zipformer2 audio tagger",
        "url": "https://github.com/k2-fsa/icefall/tree/master/egs/audioset/AT/zipformer",
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=filename, meta_data=meta_data)


def optimize_model(filename):
    # see
    # https://github.com/microsoft/onnxruntime/issues/1899#issuecomment-534806537
    # and
    # https://github.com/onnx/onnx/issues/582#issuecomment-937788108
    # and
    # https://github.com/onnx/optimizer/issues/110
    # and
    # https://qiita.com/Yossy_Hal/items/34f3b2aef2199baf7f5f
    passes = ["eliminate_unused_initializer"]
    onnx_model = onnx.load(filename)
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)

    model_simp, check = simplify(onnx_model)
    if check:
        logging.info("Simplified the model!")
        onnx_model = model_simp
    else:
        logging.info("Failed to simplify the model!")

    onnx.save(onnx_model, filename)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    model.to(device)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to("cpu")
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    model = OnnxAudioTagger(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        classifier=model.classifier,
    )

    model_num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"total parameters: {model_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"

    opset_version = 13

    logging.info("Exporting audio tagging model")
    model_filename = params.exp_dir / f"model-{suffix}.onnx"
    export_audio_tagging_model_onnx(
        model,
        model_filename,
        opset_version=opset_version,
    )
    optimize_model(model_filename)
    logging.info(f"Exported audio tagging model to {model_filename}")

    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

    logging.info("Generate int8 quantization models")

    model_filename_int8 = params.exp_dir / f"model-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=model_filename,
        model_output=model_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    optimize_model(model_filename_int8)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
