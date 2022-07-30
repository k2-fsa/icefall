#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This script loads ncnn models and uses them to decode waves.

./pruned_transducer_stateless3/jit_pretrained.py \
  --model-dir /path/to/ncnn/model_dir
  --bpe-model ./data/lang_bpe_500/bpe.model \
  /path/to/foo.wav \
  /path/to/bar.wav

We assume there exist following files in the given `model_dir`:

    - encoder_jit_trace.ncnn.param
    - encoder_jit_trace.ncnn.bin
    - decoder_jit_trace.ncnn.param
    - decoder_jit_trace.ncnn.bin
    - joiner_jit_trace.ncnn.param
    - joiner_jit_trace.ncnn.bin
"""

import argparse
import logging
from pathlib import Path
from typing import List

import ncnn
import torch
import torchaudio


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the ncnn models directory. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to bpe.model.""",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="Context size of the decoder model",
    )

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. "
            f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    model_dir = Path(args.model_dir)
    encoder_param = model_dir / "encoder_jit_trace.ncnn.param"
    encoder_bin = model_dir / "encoder_jit_trace.ncnn.bin"

    decoder_param = model_dir / "decoder_jit_trace.ncnn.param"
    decoder_bin = model_dir / "decoder_jit_trace.ncnn.bin"

    joiner_param = model_dir / "joiner_jit_trace.ncnn.param"
    joiner_bin = model_dir / "joiner_jit_trace.ncnn.bin"

    assert encoder_param.is_file()
    assert encoder_bin.is_file()

    assert decoder_param.is_file()
    assert decoder_bin.is_file()

    assert joiner_param.is_file()
    assert joiner_bin.is_file()

    encoder = ncnn.Net()
    decoder = ncnn.Net()
    joiner = ncnn.Net()

    #  encoder.load_param(str(encoder_param)) # not working yet
    #  decoder.load_param(str(decoder_param))
    joiner.load_param(str(joiner_param))

    encoder.clear()
    decoder.clear()
    joiner.clear()

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
