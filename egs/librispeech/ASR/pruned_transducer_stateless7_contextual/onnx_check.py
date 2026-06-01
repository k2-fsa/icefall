#!/usr/bin/env python3
#
# Copyright 2022 Xiaomi Corporation (Author: Fangjun Kuang)
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
This script checks that exported onnx models produce the same output
with the given torchscript model for the same input.

We use the pre-trained model from
https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained-iter-1224000-avg-14.pt"

cd exp
ln -s pretrained-iter-1224000-avg-14.pt epoch-9999.pt
popd

2. Export the model via torchscript (torch.jit.script())

./pruned_transducer_stateless3/export.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 9999 \
  --avg 1 \
  --exp-dir $repo/exp/ \
  --jit 1

It will generate the following file in $repo/exp:
    - cpu_jit.pt

3. Export the model to ONNX

./pruned_transducer_stateless3/export-onnx.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 9999 \
  --avg 1 \
  --exp-dir $repo/exp/

It will generate the following 3 files inside $repo/exp:

  - encoder-epoch-9999-avg-1.onnx
  - decoder-epoch-9999-avg-1.onnx
  - joiner-epoch-9999-avg-1.onnx

4. Run this file

./pruned_transducer_stateless3/onnx_check.py \
  --jit-filename $repo/exp/cpu_jit.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-9999-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-9999-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-9999-avg-1.onnx
"""

import argparse
import logging

from icefall import is_module_available
from onnx_pretrained import OnnxModel

import torch


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--jit-filename",
        required=True,
        type=str,
        help="Path to the torchscript model",
    )

    parser.add_argument(
        "--onnx-encoder-filename",
        required=True,
        type=str,
        help="Path to the onnx encoder model",
    )

    parser.add_argument(
        "--onnx-decoder-filename",
        required=True,
        type=str,
        help="Path to the onnx decoder model",
    )

    parser.add_argument(
        "--onnx-joiner-filename",
        required=True,
        type=str,
        help="Path to the onnx joiner model",
    )

    return parser


def test_encoder(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    C = 80
    for i in range(3):
        N = torch.randint(low=1, high=20, size=(1,)).item()
        T = torch.randint(low=30, high=50, size=(1,)).item()
        logging.info(f"test_encoder: iter {i}, N={N}, T={T}")

        x = torch.rand(N, T, C)
        x_lens = torch.randint(low=30, high=T + 1, size=(N,))
        x_lens[0] = T

        torch_encoder_out, torch_encoder_out_lens = torch_model.encoder(x, x_lens)
        torch_encoder_out = torch_model.joiner.encoder_proj(torch_encoder_out)

        onnx_encoder_out, onnx_encoder_out_lens = onnx_model.run_encoder(x, x_lens)

        assert torch.allclose(torch_encoder_out, onnx_encoder_out, atol=1e-05), (
            (torch_encoder_out - onnx_encoder_out).abs().max()
        )


def test_decoder(
    torch_model: torch.jit.ScriptModule,
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
        torch_decoder_out = torch_model.decoder(x, need_pad=torch.tensor([False]))
        torch_decoder_out = torch_model.joiner.decoder_proj(torch_decoder_out)
        torch_decoder_out = torch_decoder_out.squeeze(1)

        onnx_decoder_out = onnx_model.run_decoder(x)
        assert torch.allclose(torch_decoder_out, onnx_decoder_out, atol=1e-4), (
            (torch_decoder_out - onnx_decoder_out).abs().max()
        )


def test_joiner(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    encoder_dim = torch_model.joiner.encoder_proj.weight.shape[1]
    decoder_dim = torch_model.joiner.decoder_proj.weight.shape[1]
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_joiner: iter {i}, N={N}")
        encoder_out = torch.rand(N, encoder_dim)
        decoder_out = torch.rand(N, decoder_dim)

        projected_encoder_out = torch_model.joiner.encoder_proj(encoder_out)
        projected_decoder_out = torch_model.joiner.decoder_proj(decoder_out)

        torch_joiner_out = torch_model.joiner(encoder_out, decoder_out)
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

    torch_model = torch.jit.load(args.jit_filename)

    onnx_model = OnnxModel(
        encoder_model_filename=args.onnx_encoder_filename,
        decoder_model_filename=args.onnx_decoder_filename,
        joiner_model_filename=args.onnx_joiner_filename,
    )

    logging.info("Test encoder")
    test_encoder(torch_model, onnx_model)

    logging.info("Test decoder")
    test_decoder(torch_model, onnx_model)

    logging.info("Test joiner")
    test_joiner(torch_model, onnx_model)
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
    torch.manual_seed(20220727)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
