#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corp. (Author: Yifan Yang)
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
Usage:
./pruned_transducer_stateless7/compute_ppl.py \
    --ngram-lm-path ./download/lm/3gram_pruned_1e7.arpa

"""


import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple

import kenlm
import torch
from asr_datamodule import GigaSpeechAsrDataModule


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ngram-lm-path",
        type=str,
        default="download/lm/3gram_pruned_1e7.arpa",
        help="The lang dir containing word table and LG graph",
    )

    return parser


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    model: kenlm.Model,
) -> Dict[str, float]:
    """
    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      model:
        A ngram lm of kenlm.Model object.
    Returns:
      Return the perplexity of the giving dataset.
    """
    sum_score_log = 0
    sum_n = 0
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        for text in texts:
            sum_n += len(text.split()) + 1
            sum_score_log += -1 * model.score(text)

    ppl = math.pow(10.0, sum_score_log / sum_n)

    return ppl


def main():
    parser = get_parser()
    GigaSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    logging.info("About to load ngram LM")
    model = kenlm.Model(args.ngram_lm_path)

    gigaspeech = GigaSpeechAsrDataModule(args)

    dev_cuts = gigaspeech.dev_cuts()
    test_cuts = gigaspeech.test_cuts()

    dev_dl = gigaspeech.test_dataloaders(dev_cuts)
    test_dl = gigaspeech.test_dataloaders(test_cuts)

    test_sets = ["dev", "test"]
    test_dls = [dev_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        ppl = decode_dataset(
            dl=test_dl,
            model=model,
        )
        logging.info(f"{test_set} PPL: {ppl}")

    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
