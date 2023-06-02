#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang)

"""
This script checks that exported ONNX models produce the same output
with the given torchscript model for the same input.

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained.pt"
cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model via torch.jit.trace()

./pruned_transducer_stateless7_streaming/jit_trace_export.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

It will generate the following 3 files inside $repo/exp

  - encoder_jit_trace.pt
  - decoder_jit_trace.pt
  - joiner_jit_trace.pt

3. Export the model to ONNX

./pruned_transducer_stateless7_streaming/export-onnx.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

It will generate the following 3 files inside $repo/exp:

  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx

4. Run this file

./pruned_transducer_stateless7_streaming/onnx_check.py \
  --jit-encoder-filename $repo/exp/encoder_jit_trace.pt \
  --jit-decoder-filename $repo/exp/decoder_jit_trace.pt \
  --jit-joiner-filename $repo/exp/joiner_jit_trace.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-99-avg-1.onnx
"""

import argparse
import logging

import torch
from onnx_pretrained import OnnxModel
from zipformer import stack_states

from icefall import is_module_available


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--jit-encoder-filename",
        required=True,
        type=str,
        help="Path to the torchscript encoder model",
    )

    parser.add_argument(
        "--jit-decoder-filename",
        required=True,
        type=str,
        help="Path to the torchscript decoder model",
    )

    parser.add_argument(
        "--jit-joiner-filename",
        required=True,
        type=str,
        help="Path to the torchscript joiner model",
    )

    parser.add_argument(
        "--onnx-encoder-filename",
        required=True,
        type=str,
        help="Path to the ONNX encoder model",
    )

    parser.add_argument(
        "--onnx-decoder-filename",
        required=True,
        type=str,
        help="Path to the ONNX decoder model",
    )

    parser.add_argument(
        "--onnx-joiner-filename",
        required=True,
        type=str,
        help="Path to the ONNX joiner model",
    )

    return parser


def test_encoder(
    torch_encoder_model: torch.jit.ScriptModule,
    torch_encoder_proj_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    N = torch.randint(1, 100, size=(1,)).item()
    T = onnx_model.segment
    C = 80
    x_lens = torch.tensor([T] * N)
    torch_states = [torch_encoder_model.get_init_state() for _ in range(N)]
    torch_states = stack_states(torch_states)

    onnx_model.init_encoder_states(N)

    for i in range(5):
        logging.info(f"test_encoder: iter {i}")
        x = torch.rand(N, T, C)
        torch_encoder_out, _, torch_states = torch_encoder_model(
            x, x_lens, torch_states
        )
        torch_encoder_out = torch_encoder_proj_model(torch_encoder_out)

        onnx_encoder_out = onnx_model.run_encoder(x)

        assert torch.allclose(torch_encoder_out, onnx_encoder_out, atol=1e-4), (
            (torch_encoder_out - onnx_encoder_out).abs().max()
        )


def test_decoder(
    torch_decoder_model: torch.jit.ScriptModule,
    torch_decoder_proj_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    context_size = onnx_model.context_size
    vocab_size = onnx_model.vocab_size
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_decoder: iter {i}, N={N}")
        x = torch.randint(
            low=1,
            high=vocab_size,
            size=(N, context_size),
            dtype=torch.int64,
        )
        torch_decoder_out = torch_decoder_model(x, need_pad=torch.tensor([False]))
        torch_decoder_out = torch_decoder_proj_model(torch_decoder_out)
        torch_decoder_out = torch_decoder_out.squeeze(1)

        onnx_decoder_out = onnx_model.run_decoder(x)
        assert torch.allclose(torch_decoder_out, onnx_decoder_out, atol=1e-4), (
            (torch_decoder_out - onnx_decoder_out).abs().max()
        )


def test_joiner(
    torch_joiner_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    encoder_dim = torch_joiner_model.encoder_proj.weight.shape[1]
    decoder_dim = torch_joiner_model.decoder_proj.weight.shape[1]
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_joiner: iter {i}, N={N}")
        encoder_out = torch.rand(N, encoder_dim)
        decoder_out = torch.rand(N, decoder_dim)

        projected_encoder_out = torch_joiner_model.encoder_proj(encoder_out)
        projected_decoder_out = torch_joiner_model.decoder_proj(decoder_out)

        torch_joiner_out = torch_joiner_model(encoder_out, decoder_out)
        onnx_joiner_out = onnx_model.run_joiner(
            projected_encoder_out, projected_decoder_out
        )

        assert torch.allclose(torch_joiner_out, onnx_joiner_out, atol=1e-4), (
            (torch_joiner_out - onnx_joiner_out).abs().max()
        )


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    torch_encoder_model = torch.jit.load(args.jit_encoder_filename)
    torch_decoder_model = torch.jit.load(args.jit_decoder_filename)
    torch_joiner_model = torch.jit.load(args.jit_joiner_filename)

    onnx_model = OnnxModel(
        encoder_model_filename=args.onnx_encoder_filename,
        decoder_model_filename=args.onnx_decoder_filename,
        joiner_model_filename=args.onnx_joiner_filename,
    )

    logging.info("Test encoder")
    # When exporting the model to onnx, we have already put the encoder_proj
    # inside the encoder.
    test_encoder(torch_encoder_model, torch_joiner_model.encoder_proj, onnx_model)

    logging.info("Test decoder")
    # When exporting the model to onnx, we have already put the decoder_proj
    # inside the decoder.
    test_decoder(torch_decoder_model, torch_joiner_model.decoder_proj, onnx_model)

    logging.info("Test joiner")
    test_joiner(torch_joiner_model, onnx_model)

    logging.info("Finished checking ONNX models")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    torch.manual_seed(20230207)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
