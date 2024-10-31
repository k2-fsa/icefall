#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao)
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
This file computes fbank features of the LJSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from lhotse import CutSet, LilcomChunkyWriter, load_manifest
from lhotse.audio import RecordingSet
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.supervision import SupervisionSet
from lhotse.utils import Seconds, compute_num_frames
from matcha.audio import mel_spectrogram

from icefall.utils import get_executor


@dataclass
class MyFbankConfig:
    n_fft: int
    n_mels: int
    sampling_rate: int
    hop_length: int
    win_length: int
    f_min: float
    f_max: float


@register_extractor
class MyFbank(FeatureExtractor):

    name = "MyFbank"
    config_type = MyFbankConfig

    def __init__(self, config):
        super().__init__(config=config)

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def extract(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> torch.Tensor:
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        samples = torch.from_numpy(samples)
        assert samples.ndim == 2, samples.shape
        assert samples.shape[0] == 1, samples.shape

        mel = (
            mel_spectrogram(
                samples,
                self.config.n_fft,
                self.config.n_mels,
                self.config.sampling_rate,
                self.config.hop_length,
                self.config.win_length,
                self.config.f_min,
                self.config.f_max,
                center=False,
            )
            .squeeze()
            .t()
        )

        assert mel.ndim == 2, mel.shape
        assert mel.shape[1] == self.config.n_mels, mel.shape

        num_frames = compute_num_frames(
            samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate
        )

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)

        return mel.numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.sampling_rate


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )
    return parser


def compute_fbank_ljspeech(num_jobs: int):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    if num_jobs < 1:
        num_jobs = os.cpu_count()

    logging.info(f"num_jobs: {num_jobs}")
    logging.info(f"src_dir: {src_dir}")
    logging.info(f"output_dir: {output_dir}")
    config = MyFbankConfig(
        n_fft=1024,
        n_mels=80,
        sampling_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
    )

    prefix = "ljspeech"
    suffix = "jsonl.gz"
    partition = "all"

    recordings = load_manifest(
        src_dir / f"{prefix}_recordings_{partition}.{suffix}", RecordingSet
    )
    supervisions = load_manifest(
        src_dir / f"{prefix}_supervisions_{partition}.{suffix}", SupervisionSet
    )

    extractor = MyFbank(config)

    with get_executor() as ex:  # Initialize the executor only once.
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{cuts_filename} already exists - skipping.")
            return
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=recordings, supervisions=supervisions
        )

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    # Torch's multithreaded behavior needs to be disabled or
    # it wastes a lot of CPU and slow things down.
    # Do this outside of main() in case it needs to take effect
    # even when we are not invoking the main (e.g. when spawning subprocesses).
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_parser().parse_args()
    compute_fbank_ljspeech(args.num_jobs)
