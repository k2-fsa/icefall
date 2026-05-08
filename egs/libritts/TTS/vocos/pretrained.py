#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Wei Kang)
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
This script loads a checkpoint and uses it to decode waves.
You can generate the checkpoint with the following command:
"""


import argparse
import logging
import math
from pathlib import Path
from typing import List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_model, get_params
from lhotse import Fbank, FbankConfig

from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
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

    add_model_arguments(parser)

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
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    params.device = device

    output_dir = Path(params.checkpoint).parent / params.output_dir
    output_dir.mkdir(exist_ok=True)
    params.output_dir = output_dir

    logging.info(f"{params}")

    logging.info("Creating model")
    model = get_model(params)

    model = model.generator

    checkpoint = torch.load(params.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")

    config = FbankConfig(
        sampling_rate=params.sampling_rate,
        frame_length=params.frame_length / params.sampling_rate,  # (in second),
        frame_shift=params.frame_shift / params.sampling_rate,  # (in second)
        use_fft_mag=params.use_fft_mag,
    )
    fbank = Fbank(config)

    logging.info(f"Reading sound files: {params.sound_files}")

    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sampling_rate
    )
    wave_lengths = [w.size(0) for w in waves]
    waves = pad_sequence(waves, batch_first=True, padding_value=0)

    features = (
        fbank.extract_batch(waves, sampling_rate=params.sampling_rate)
        .permute(0, 2, 1)
        .to(device)
    )

    logging.info("Generating started")

    # model forward
    audios = model(features)

    for i, filename in enumerate(params.sound_files):
        audio = audios[i : i + 1, 0 : wave_lengths[i]]
        ofilename = params.output_dir / filename.split("/")[-1]
        logging.info(f"Writting audio : {ofilename}")
        torchaudio.save(str(ofilename), audio.cpu(), params.sampling_rate)

    logging.info("Generating Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
