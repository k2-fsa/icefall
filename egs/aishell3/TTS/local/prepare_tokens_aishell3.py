#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
This file reads the texts in given manifest and save the new cuts with tokens.
"""

import logging
from pathlib import Path

from lhotse import CutSet, load_manifest

from tokenizer import Tokenizer


def prepare_tokens_aishell3():
    output_dir = Path("data/spectrogram")
    prefix = "aishell3"
    suffix = "jsonl.gz"
    partitions = ("train", "test")

    tokenizer = Tokenizer()

    for partition in partitions:
        cut_set = load_manifest(output_dir / f"{prefix}_cuts_{partition}.{suffix}")

        new_cuts = []
        i = 0
        for cut in cut_set:
            # Each cut only contains one supervision
            assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
            text = cut.supervisions[0].text
            cut.tokens = tokenizer.text_to_tokens(text)

            new_cuts.append(cut)

        new_cut_set = CutSet.from_cuts(new_cuts)
        new_cut_set.to_file(
            output_dir / f"{prefix}_cuts_with_tokens_{partition}.{suffix}"
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    prepare_tokens_aishell3()
