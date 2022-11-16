#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

import logging
from pathlib import Path

import sentencepiece as spm
import torch


def main():
    lm_training_data = Path("./data/bpe_500/lm_data.pt")
    bpe_model = Path("./data/bpe_500/bpe.model")
    if not lm_training_data.exists():
        logging.warning(f"{lm_training_data} does not exist - skipping")
        return

    if not bpe_model.exists():
        logging.warning(f"{bpe_model} does not exist - skipping")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_model))

    data = torch.load(lm_training_data)
    words2bpe = data["words"]
    sentences = data["sentences"]

    ss = []
    unk = sp.decode(sp.unk_id()).strip()
    for i in range(10):
        s = sp.decode(words2bpe[sentences[i]].values.tolist())
        s = s.replace(unk, "<unk>")
        ss.append(s)

    for s in ss:
        print(s)
    # You can compare the output with the first 10 lines of ptb.train.txt


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
