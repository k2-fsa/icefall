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
This script loads ONNX models and uses them to decode waves.
You can use the following command to get the exported models:

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model to ONNX

./zipformer/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --causal False

It will generate the following 3 files inside $repo/exp:

  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx

3. Run this file

./zipformer/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav
"""

import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple

import onnxruntime as ort
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from lhotse import Fbank, FbankConfig
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="The sampleing rate of libritts dataset",
    )

    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        help="Frame shift.",
    )

    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        help="Frame shift.",
    )

    parser.add_argument(
        "--use-fft-mag",
        type=str2bool,
        default=True,
        help="Whether to use magnitude of fbank, false to use power energy.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_audios",
        help="The generated will be written to.",
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

    return parser


class OnnxModel:
    def __init__(
        self,
        model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_model(model_filename)

    def init_model(self, model_filename: str):
        self.model = ort.InferenceSession(
            model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def run_model(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 2-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a tuple containing:
            - encoder_out, its shape is (N, T', joiner_dim)
            - encoder_out_lens, its shape is (N,)
        """
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )
        return torch.from_numpy(out[0])


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
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    output_dir = Path(args.model_filename).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    args.output_dir = output_dir
    logging.info(vars(args))

    model = OnnxModel(model_filename=args.model_filename)

    config = FbankConfig(
        sampling_rate=args.sampling_rate,
        frame_length=args.frame_length / args.sampling_rate,  # (in second),
        frame_shift=args.frame_shift / args.sampling_rate,  # (in second)
        use_fft_mag=args.use_fft_mag,
    )
    fbank = Fbank(config)

    logging.info(f"Reading sound files: {args.sound_files}")

    waves = read_sound_files(
        filenames=args.sound_files, expected_sample_rate=args.sampling_rate
    )
    wave_lengths = [w.size(0) for w in waves]
    waves = pad_sequence(waves, batch_first=True, padding_value=0)

    logging.info(f"waves : {waves.shape}")

    features = fbank.extract_batch(waves, sampling_rate=args.sampling_rate)

    if features.dim() == 2:
        features = features.unsqueeze(0)

    features = features.permute(0, 2, 1)

    logging.info(f"features : {features.shape}")

    logging.info("Generating started")

    # model forward
    audios = model.run_model(features)

    for i, filename in enumerate(args.sound_files):
        audio = audios[i : i + 1, 0 : wave_lengths[i]]
        ofilename = args.output_dir / filename.split("/")[-1]
        logging.info(f"Writting audio : {ofilename}")
        torchaudio.save(str(ofilename), audio.cpu(), args.sampling_rate)

    logging.info("Generating Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
