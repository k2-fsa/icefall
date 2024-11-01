#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                       Zengrui Jin,)
#                   2024  Tsinghua University (authors: Zengrui Jin,)
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
This file reads the texts in given manifest and save the new cuts with phoneme tokens.
"""

import logging
from pathlib import Path

import tacotron_cleaner.cleaners
from lhotse import CutSet, load_manifest
from piper_phonemize import phonemize_espeak
from tqdm.auto import tqdm


def remove_punc_to_upper(text: str) -> str:
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")
    s_list = [x.upper() if x in tokens else " " for x in text]
    s = " ".join("".join(s_list).split()).strip()
    return s


def prepare_tokens_libritts():
    output_dir = Path("data/spectrogram")
    prefix = "libritts"
    suffix = "jsonl.gz"
    partitions = (
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-all-shuf",
        "train-clean-460",
        # "train-clean-100",
        # "train-clean-360",
        # "train-other-500",
    )

    for partition in partitions:
        cut_set = load_manifest(output_dir / f"{prefix}_cuts_{partition}.{suffix}")

        new_cuts = []
        for cut in tqdm(cut_set):
            # Each cut only contains one supervision
            assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
            text = cut.supervisions[0].text
            # Text normalization
            text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            # Convert to phonemes
            tokens_list = phonemize_espeak(text, "en-us")
            tokens = []
            for t in tokens_list:
                tokens.extend(t)
            cut.tokens = tokens
            cut.supervisions[0].normalized_text = remove_punc_to_upper(text)

            new_cuts.append(cut)

        new_cut_set = CutSet.from_cuts(new_cuts)
        new_cut_set.to_file(
            output_dir / f"{prefix}_cuts_with_tokens_{partition}.{suffix}"
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    prepare_tokens_libritts()
