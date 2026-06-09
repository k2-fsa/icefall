#!/usr/bin/env python3
#
# Copyright 2023-2026 Xiaomi Corporation (Author: Fangjun Kuang, Wei Kang)
# Copyright 2023 Danqing Fu (danqing.fu@gmail.com)
# 
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script exports a streaming transducer model from PyTorch to ONNX.

Usage:

cd egs/librispeech/ASR

./zapformer/export-onnx-streaming.py \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 9 \
  --avg 2 \
  --exp-dir zapformer/exp \
  --causal 1 \
  --chunk-size 32 \
  --left-context-frames 128

It will generate the following 3 files inside exp-dir:

  - encoder-epoch-9-avg-2-chunk-32-left-128.onnx
  - decoder-epoch-9-avg-2-chunk-32-left-128.onnx
  - joiner-epoch-9-avg-2-chunk-32-left-128.onnx
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import k2
import onnx
import torch
import torch.nn as nn
from decoder import Decoder
from onnxruntime.quantization import QuantType, quantize_dynamic
from train import add_model_arguments, get_model, get_params
from zapformer import Zapformer

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
        "--dynamic-batch",
        type=int,
        default=1,
        help="1 to support dynamic batch size. 0 to support only batch size == 1",
    )

    parser.add_argument(
        "--enable-int8-quantization",
        type=int,
        default=1,
        help="1 to also export int8 onnx models.",
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
        default="zapformer/exp",
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

    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=False,
        help="Whether to export models in fp16",
    )

    add_model_arguments(parser)

    return parser


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    import onnxmltools
    from onnxmltools.utils.float16_converter import convert_float_to_float16

    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


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


class OnnxEncoder(nn.Module):
    """A wrapper for Zapformer and the encoder_proj from the joiner"""

    def __init__(
        self, encoder: Zapformer, encoder_embed: nn.Module, encoder_proj: nn.Linear
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.encoder_proj = encoder_proj
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]

    def forward(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        N = x.size(0)
        T = self.chunk_size * 2 + 7
        x_lens = torch.tensor([T] * N, device=x.device)
        left_context_len = self.left_context_len

        embed_cache = states[-2]
        x, x_lens, new_embed_cache = self.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cache=embed_cache,
        )
        assert x.size(1) == self.chunk_size, (x.size(1), self.chunk_size)

        src_key_padding_mask = torch.zeros(N, self.chunk_size, dtype=torch.bool)

        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]  # (batch,)
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        new_processed_lens = processed_lens + x_lens
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)
        encoder_caches = states[:-2]
        logging.info(f"len_encoder_caches={len(encoder_caches)}")
        (
            encoder_out,
            encoder_out_lens,
            new_encoder_caches,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            caches=encoder_caches,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.encoder_proj(encoder_out)

        new_states = new_encoder_caches + [
            new_embed_cache,
            new_processed_lens,
        ]

        return encoder_out, new_states

    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Returns a list of cached tensors of all encoder layers. For layer-i,
        states[i*9:(i+1)*9] is (cached_key, cached_value, cached_conv,
        cached_norm_stats, cached_norm_len, cached_attn_wm_sum,
        cached_attn_wm_num_frames, cached_conv_wm_sum, cached_conv_wm_num_frames).
        states[-2] is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs).
        states[-1] is processed_lens of shape (batch,).
        """
        states = self.encoder.get_init_caches(batch_size, device)

        embed_cache = self.encoder_embed.get_init_cache(batch_size, device)
        states.append(embed_cache)

        processed_lens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        states.append(processed_lens)

        return states


class OnnxDecoder(nn.Module):
    """A wrapper for Decoder and the decoder_proj from the joiner"""

    def __init__(self, decoder: Decoder, decoder_proj: nn.Linear):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        need_pad = False
        decoder_output = self.decoder(y, need_pad=need_pad)
        decoder_output = decoder_output.squeeze(1)
        output = self.decoder_proj(decoder_output)

        return output


class OnnxJoiner(nn.Module):
    """A wrapper for the joiner"""

    def __init__(self, output_linear: nn.Linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        logit = encoder_out + decoder_out
        logit = 2.0 * self.output_linear(torch.tanh(logit))
        return logit


def export_encoder_model_onnx(
    encoder_model: OnnxEncoder,
    encoder_filename: str,
    opset_version: int = 11,
    feature_dim: int = 80,
    dynamic_batch: bool = True,
) -> None:
    encoder_model.encoder.__class__.forward = (
        encoder_model.encoder.__class__.streaming_forward
    )

    decode_chunk_len = encoder_model.chunk_size * 2
    T = decode_chunk_len + 7

    x = torch.rand(1, T, feature_dim, dtype=torch.float32)
    init_state = encoder_model.get_init_states()
    logging.info(f"len(init_state): {len(init_state)}")

    # Warm up angular freq bases for tracing
    left_context_len = encoder_model.left_context_len
    ds_factors = encoder_model.encoder.downsampling_factor
    max_seq_len = left_context_len + encoder_model.chunk_size
    encoder_model.encoder.warmup_angular_freq_bases(
        seq_len=max_seq_len, left_context_len=left_context_len, device=x.device
    )

    inputs = {}
    input_names = ["x"]

    outputs = {}
    output_names = ["encoder_out"]

    # Count total number of layers across all encoder stacks
    total_layers = sum(encoder_model.encoder.num_encoder_layers)
    logging.info(f"total encoder layers: {total_layers}")

    def build_inputs_outputs(tensors, i):
        assert len(tensors) == 9, len(tensors)

        # (downsample_left, batch_size, key_dim)
        name = f"cached_key_{i}"
        logging.info(f"{name}.shape: {tensors[0].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (downsample_left, batch_size, value_dim)
        name = f"cached_value_{i}"
        logging.info(f"{name}.shape: {tensors[1].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # (batch_size, embed_dim, conv_left_pad)
        name = f"cached_conv_{i}"
        logging.info(f"{name}.shape: {tensors[2].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # cached_norm_stats: (batch_size,)
        name = f"cached_norm_stats_{i}"
        logging.info(f"{name}.shape: {tensors[3].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # cached_norm_len: (batch_size,)
        name = f"cached_norm_len_{i}"
        logging.info(f"{name}.shape: {tensors[4].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # cached_attn_wm_sum: (1, batch_size, attn_value_dim)
        name = f"cached_attn_wm_sum_{i}"
        logging.info(f"{name}.shape: {tensors[5].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # cached_attn_wm_num_frames: (batch_size,)
        name = f"cached_attn_wm_num_frames_{i}"
        logging.info(f"{name}.shape: {tensors[6].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # cached_conv_wm_sum: (1, batch_size, embed_dim)
        name = f"cached_conv_wm_sum_{i}"
        logging.info(f"{name}.shape: {tensors[7].shape}")
        inputs[name] = {1: "N"}
        outputs[f"new_{name}"] = {1: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

        # cached_conv_wm_num_frames: (batch_size,)
        name = f"cached_conv_wm_num_frames_{i}"
        logging.info(f"{name}.shape: {tensors[8].shape}")
        inputs[name] = {0: "N"}
        outputs[f"new_{name}"] = {0: "N"}
        input_names.append(name)
        output_names.append(f"new_{name}")

    num_encoder_layers = encoder_model.encoder.num_encoder_layers
    encoder_dims = encoder_model.encoder.encoder_dim
    conv_params = encoder_model.encoder.conv_params
    ds = encoder_model.encoder.downsampling_factor
    left_context_len_per_stack = [left_context_len // k for k in ds]
    query_head_dims = encoder_model.encoder.query_head_dim
    value_head_dims = encoder_model.encoder.value_head_dim
    num_heads = encoder_model.encoder.num_heads

    meta_data = {
        "model_type": "zapformer",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "streaming zapformer",
        "decode_chunk_len": str(decode_chunk_len),
        "T": str(T),
        "num_encoder_layers": ",".join(map(str, num_encoder_layers)),
        "encoder_dims": ",".join(map(str, encoder_dims)),
        "conv_params": ",".join(map(str, conv_params)),
        "left_context_len": ",".join(map(str, left_context_len_per_stack)),
        "query_head_dims": ",".join(map(str, query_head_dims)),
        "value_head_dims": ",".join(map(str, value_head_dims)),
        "num_heads": ",".join(map(str, num_heads)),
    }

    logging.info(f"meta_data: {meta_data}")

    # 9 tensors per layer
    for i in range(len(init_state[:-2]) // 9):
        build_inputs_outputs(init_state[i * 9 : (i + 1) * 9], i)

    # (batch_size, channels, left_pad, freq)
    embed_cache = init_state[-2]
    name = "embed_cache"
    logging.info(f"{name}.shape: {embed_cache.shape}")
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

    logging.info(f"input_names: {input_names}")
    logging.info(f"output_names: {output_names}")

    torch.onnx.export(
        encoder_model,
        (x, init_state),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "x": {0: "N"},
            "encoder_out": {0: "N"},
            **inputs,
            **outputs,
        }
        if dynamic_batch
        else {},
    )

    add_meta_data(filename=encoder_filename, meta_data=meta_data)


def export_decoder_model_onnx(
    decoder_model: OnnxDecoder,
    decoder_filename: str,
    opset_version: int = 11,
    dynamic_batch: bool = True,
) -> None:
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size

    y = torch.zeros(1, context_size, dtype=torch.int64)
    decoder_model = torch.jit.script(decoder_model)
    torch.onnx.export(
        decoder_model,
        y,
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        }
        if dynamic_batch
        else {},
    )

    meta_data = {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    }
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def export_joiner_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
    dynamic_batch: bool = True,
) -> None:
    joiner_dim = joiner_model.output_linear.weight.shape[1]
    logging.info(f"joiner dim: {joiner_dim}")

    projected_encoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)

    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "encoder_out",
            "decoder_out",
        ],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        }
        if dynamic_batch
        else {},
    )
    meta_data = {
        "joiner_dim": str(joiner_dim),
    }
    add_meta_data(filename=joiner_filename, meta_data=meta_data)


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
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=False
            )
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
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=False
            )
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
                ),
                strict=False,
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
                ),
                strict=False,
            )

    model.to("cpu")
    model.eval()

    encoder = OnnxEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    )

    decoder = OnnxDecoder(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )

    joiner = OnnxJoiner(output_linear=model.joiner.output_linear)

    encoder_num_param = sum([p.numel() for p in encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in decoder.parameters()])
    joiner_num_param = sum([p.numel() for p in joiner.parameters()])
    total_num_param = encoder_num_param + decoder_num_param + joiner_num_param
    logging.info(f"encoder parameters: {encoder_num_param}")
    logging.info(f"decoder parameters: {decoder_num_param}")
    logging.info(f"joiner parameters: {joiner_num_param}")
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"
    suffix += f"-chunk-{params.chunk_size}"
    suffix += f"-left-{params.left_context_frames}"

    opset_version = 13

    logging.info("Exporting encoder")
    encoder_filename = params.exp_dir / f"encoder-{suffix}.onnx"
    export_encoder_model_onnx(
        encoder,
        str(encoder_filename),
        opset_version=opset_version,
        feature_dim=params.feature_dim,
        dynamic_batch=params.dynamic_batch == 1,
    )
    logging.info(f"Exported encoder to {encoder_filename}")

    logging.info("Exporting decoder")
    decoder_filename = params.exp_dir / f"decoder-{suffix}.onnx"
    export_decoder_model_onnx(
        decoder,
        str(decoder_filename),
        opset_version=opset_version,
        dynamic_batch=params.dynamic_batch == 1,
    )
    logging.info(f"Exported decoder to {decoder_filename}")

    logging.info("Exporting joiner")
    joiner_filename = params.exp_dir / f"joiner-{suffix}.onnx"
    export_joiner_model_onnx(
        joiner,
        str(joiner_filename),
        opset_version=opset_version,
        dynamic_batch=params.dynamic_batch == 1,
    )
    logging.info(f"Exported joiner to {joiner_filename}")

    if params.fp16:
        logging.info("Generate fp16 models")

        encoder_filename_fp16 = params.exp_dir / f"encoder-{suffix}.fp16.onnx"
        export_onnx_fp16(encoder_filename, encoder_filename_fp16)

        decoder_filename_fp16 = params.exp_dir / f"decoder-{suffix}.fp16.onnx"
        export_onnx_fp16(decoder_filename, decoder_filename_fp16)

        joiner_filename_fp16 = params.exp_dir / f"joiner-{suffix}.fp16.onnx"
        export_onnx_fp16(joiner_filename, joiner_filename_fp16)

    # Generate int8 quantization models
    if params.enable_int8_quantization:
        logging.info("Generate int8 quantization models")

        encoder_filename_int8 = params.exp_dir / f"encoder-{suffix}.int8.onnx"
        quantize_dynamic(
            model_input=encoder_filename,
            model_output=encoder_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

        decoder_filename_int8 = params.exp_dir / f"decoder-{suffix}.int8.onnx"
        quantize_dynamic(
            model_input=decoder_filename,
            model_output=decoder_filename_int8,
            op_types_to_quantize=["MatMul", "Gather"],
            weight_type=QuantType.QInt8,
        )

        joiner_filename_int8 = params.exp_dir / f"joiner-{suffix}.int8.onnx"
        quantize_dynamic(
            model_input=joiner_filename,
            model_output=joiner_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
