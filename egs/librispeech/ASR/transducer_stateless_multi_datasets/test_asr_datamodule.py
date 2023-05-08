#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./transducer_stateless_multi_datasets/test_asr_datamodule.py
"""

import argparse
import random
from pathlib import Path

from asr_datamodule import AsrDataModule
from gigaspeech import GigaSpeech
from lhotse import load_manifest
from librispeech import LibriSpeech


def test_dataset():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    print(args)

    if args.enable_musan:
        cuts_musan = load_manifest(Path(args.manifest_dir) / "musan_cuts.jsonl.gz")
    else:
        cuts_musan = None

    librispeech = LibriSpeech(manifest_dir=args.manifest_dir)
    gigaspeech = GigaSpeech(manifest_dir=args.manifest_dir)

    train_clean_100 = librispeech.train_clean_100_cuts()
    train_S = gigaspeech.train_S_cuts()

    asr_datamodule = AsrDataModule(args)

    libri_train_dl = asr_datamodule.train_dataloaders(
        train_clean_100,
        on_the_fly_feats=False,
        cuts_musan=cuts_musan,
    )

    giga_train_dl = asr_datamodule.train_dataloaders(
        train_S,
        on_the_fly_feats=True,
        cuts_musan=cuts_musan,
    )

    seed = 20220216
    rng = random.Random(seed)

    for epoch in range(2):
        print("epoch", epoch)
        batch_idx = 0
        libri_train_dl.sampler.set_epoch(epoch)
        giga_train_dl.sampler.set_epoch(epoch)

        iter_libri = iter(libri_train_dl)
        iter_giga = iter(giga_train_dl)
        while True:
            idx = rng.choices((0, 1), weights=[0.8, 0.2], k=1)[0]
            dl = iter_libri if idx == 0 else iter_giga
            batch_idx += 1

            print("dl idx", idx, "batch_idx", batch_idx)
            try:
                _ = next(dl)
            except StopIteration:
                print("dl idx", idx)
                print("Go to the next epoch")
                break


def main():
    test_dataset()


if __name__ == "__main__":
    main()
