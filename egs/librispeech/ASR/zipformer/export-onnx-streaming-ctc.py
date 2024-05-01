#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang, Wei Kang, Zengrui Jin)
# Copyright 2023 Danqing Fu (danqing.fu@gmail.com)

"""
This script exports a CTC model from PyTorch to ONNX.


1. Download the pre-trained streaming model with CTC head

2. Export the model to ONNX

./zipformer/export-onnx-streaming-ctc.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --num-encoder-layers "2,2,3,4,3,2" \
  --downsampling-factor "1,2,4,8,4,2" \
  --feedforward-dim "512,768,1024,1536,1024,768" \
  --num-heads "4,4,4,8,4,4" \
  --encoder-dim "192,256,384,512,384,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,256,256,256,192" \
  --cnn-module-kernel "31,31,15,15,15,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal True \
  --chunk-size 16 \
  --left-context-frames 128 \
  --use-ctc 1

The --chunk-size in training is "16,32,64,-1", so we select one of them
(excluding -1) during streaming export. The same applies to `--left-context`,
whose value is "64,128,256,-1".

It will generate the following file inside $repo/exp:

  - ctc-epoch-99-avg-1-chunk-16-left-128.onnx

See ./onnx_pretrained-streaming-ctc.py for how to use the exported ONNX models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import k2
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_model, get_params
from zipformer import Zipformer2

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import num_tokens, str2bool


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

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_bpe_500/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
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


class OnnxModel(nn.Module):
    """A wrapper for Zipformer and the ctc_head"""

    def __init__(
        self,
        encoder: Zipformer2,
        encoder_embed: nn.Module,
        ctc_output: nn.Module,
    ):
        """
        Args:
          encoder:
            A Zipformer encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
          ctc_output:
            The ctc head.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.ctc_output = ctc_output
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = 7 + 2 * 3

    def forward(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        N = x.size(0)
        T = self.chunk_size * 2 + self.pad_length
        x_lens = torch.tensor([T] * N, device=x.device)
        left_context_len = self.left_context_len

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cached_left_pad=cached_embed_left_pad,
        )
        assert x.size(1) == self.chunk_size, (x.size(1), self.chunk_size)

        src_key_padding_mask = torch.zeros(N, self.chunk_size, dtype=torch.bool)

        # processed_mask is used to mask out initial states
        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]  # (batch,)
        # (batch, left_context_size)
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        # Update processed lengths
        new_processed_lens = processed_lens + x_lens
        # (batch, left_context_size + chunk_size)
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)
        encoder_states = states[:-2]
        logging.info(f"len_encoder_states={len(encoder_states)}")
        (
            encoder_out,
            encoder_out_lens,
            new_encoder_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.ctc_output(encoder_out)
        # Now encoder_out is of shape (N, T, ctc_output_dim)

        new_states = new_encoder_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]

        return encoder_out, new_states

    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        states[-2] is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs)
        states[-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
        """
        states = self.encoder.get_init_states(batch_size, device)

        embed_states = self.encoder_embed.get_init_states(batch_size, device)

        states.append(embed_states)

        processed_lens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        states.append(processed_lens)

        return states


def export_streaming_ctc_model_onnx(
    model: OnnxModel,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    model.encoder.__class__.forward = model.encoder.__class__.streaming_forward

    decode_chunk_len = model.chunk_size * 2
    # The encoder_embed subsample features (T - 7) // 2
    # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
    T = decode_chunk_len + model.pad_length

    x = torch.rand(1, T, 80, dtype=torch.float32)
    init_state = model.get_init_states()
    num_encoders = len(model.encoder.encoder_dim)
    logging.info(f"num_encoders: {num_encoders}")
    logging.info(f"len(init_state): {len(init_state)}")

    inputs = {}
    input_names = ["x"]

    outputs = {}
    output_names = ["log_probs"]

    def build_inputs_outputs(tensors, i):
        assert len(tensors) == 6, len(tensors)

        # (downsample_left, batch_size, key_dim)
        name = f"cached_key_{i}"
        logging.info(f"{name}.shape: {tensors[0].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (1, batch_size, downsample_left, nonlin_attn_head_dim)
        name = f"cached_nonlin_attn_{i}"
        logging.info(f"{name}.shape: {tensors[1].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (downsample_left, batch_size, value_dim)
        name = f"cached_val1_{i}"
        logging.info(f"{name}.shape: {tensors[2].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (downsample_left, batch_size, value_dim)
        name = f"cached_val2_{i}"
        logging.info(f"{name}.shape: {tensors[3].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (batch_size, embed_dim, conv_left_pad)
        name = f"cached_conv1_{i}"
        logging.info(f"{name}.shape: {tensors[4].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (batch_size, embed_dim, conv_left_pad)
        name = f"cached_conv2_{i}"
        logging.info(f"{name}.shape: {tensors[5].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

    num_encoder_layers = ",".join(map(str, model.encoder.num_encoder_layers))
    encoder_dims = ",".join(map(str, model.encoder.encoder_dim))
    cnn_module_kernels = ",".join(map(str, model.encoder.cnn_module_kernel))
    ds = model.encoder.downsampling_factor
    left_context_len = model.left_context_len
    left_context_len = [left_context_len // k for k in ds]
    left_context_len = ",".join(map(str, left_context_len))
    query_head_dims = ",".join(map(str, model.encoder.query_head_dim))
    value_head_dims = ",".join(map(str, model.encoder.value_head_dim))
    num_heads = ",".join(map(str, model.encoder.num_heads))

    meta_data = {
        "model_type": "zipformer2",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "streaming ctc zipformer2",
        "decode_chunk_len": str(decode_chunk_len),  # 32
        "T": str(T),  # 32+7+2*3=45
        "num_encoder_layers": num_encoder_layers,
        "encoder_dims": encoder_dims,
        "cnn_module_kernels": cnn_module_kernels,
        "left_context_len": left_context_len,
        "query_head_dims": query_head_dims,
        "value_head_dims": value_head_dims,
        "num_heads": num_heads,
    }
    logging.info(f"meta_data: {meta_data}")

    for i in range(len(init_state[:-2]) // 6):
        build_inputs_outputs(init_state[i * 6 : (i + 1) * 6], i)

    # (batch_size, channels, left_pad, freq)
    embed_states = init_state[-2]
    name = "embed_states"
    logging.info(f"{name}.shape: {embed_states.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (batch_size,)
    processed_lens = init_state[-1]
    name = "processed_lens"
    logging.info(f"{name}.shape: {processed_lens.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    logging.info(inputs)
    logging.info(outputs)
    logging.info(input_names)
    logging.info(output_names)

    torch.onnx.export(
        model,
        (x, init_state),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "x": {0: "N"},
            "log_probs": {0: "N"},
            **inputs,
            **outputs,
        },
    )

    add_meta_data(filename=encoder_filename, meta_data=meta_data)


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

    token_table = k2.SymbolTable.from_file(params.tokens)
    params.blank_id = token_table["<blk>"]
    params.vocab_size = num_tokens(token_table) + 1

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

    convert_scaled_to_non_scaled(model, inplace=True)

    model = OnnxModel(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        ctc_output=model.ctc_output,
    )

    total_num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"
    suffix += f"-chunk-{params.chunk_size}"
    suffix += f"-left-{params.left_context_frames}"

    opset_version = 13

    logging.info("Exporting model")
    model_filename = params.exp_dir / f"ctc-{suffix}.onnx"
    export_streaming_ctc_model_onnx(
        model,
        model_filename,
        opset_version=opset_version,
    )
    logging.info(f"Exported model to {model_filename}")

    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

    logging.info("Generate int8 quantization models")

    model_filename_int8 = params.exp_dir / f"ctc-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=model_filename,
        model_output=model_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
