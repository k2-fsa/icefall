#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation (Author: Yifan Yang)
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
./tdnn/decode.py \
    --epoch 20 \
    --avg 1 \
    --exp-dir ./tdnn/exp \
    --max-duration 600
"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from tdnn import Tdnn
from asr_datamodule import SpeechCommands1DataModule
from train import get_params

from tokenizer import WakeupWordTokenizer
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="tdnn/exp",
        help="The experiment dir",
    )

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
) -> Dict[str, torch.Tensor]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]

    model_output = model(feature)
    predict = torch.argmax(model_output, 1)

    return {"predict": predict}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    tokenizer: WakeupWordTokenizer,
) -> Dict[str, int]:
    """Decode dataset.
    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      tokenizer:
        For positive samples, map their texts to corresponding token index.
        While for negative samples, map their texts to unknown no matter what they are.
    Returns:
      Return the predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    log_interval = 50

    TP = 0
    FN = 0
    FP = 0
    TN = 0
    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        batch_size = len(texts)

        results_dict = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        predict = results_dict["predict"].to("cpu")
        ref, ref_num_positive = tokenizer.texts_to_token_ids(texts)

        is_match = predict == ref
        for i in range(batch_size):
            if predict[i] in [0, 1]:
                if is_match[i]:
                    TN += 1
                else:
                    FN += 1
            else:
                if is_match[i]:
                    TP += 1
                else:
                    FP += 1

        num_cuts += batch_size

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return {"TP": TP, "FN": FN, "FP": FP, "TN": TN}


@torch.no_grad()
def main():
    parser = get_parser()
    SpeechCommands1DataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / f"decode"
    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    tokenizer = WakeupWordTokenizer(
        wakeup_words=params.wakeup_words,
        wakeup_word_tokens=params.wakeup_word_tokens,
    )

    logging.info(params)

    logging.info("About to create model")
    model = Tdnn(params.feature_dim, params.num_class)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if i >= 1:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    speechcommands1 = SpeechCommands1DataModule(args)

    test_cuts = speechcommands1.test_cuts()
    test_dl = speechcommands1.test_dataloaders(test_cuts)

    results_dict = decode_dataset(
        dl=test_dl,
        params=params,
        model=model,
        tokenizer=tokenizer,
    )

    TP = results_dict["TP"]
    FN = results_dict["FN"]
    FP = results_dict["FP"]
    TN = results_dict["TN"]
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    s = (
        f"Decoding result:\n"
        + f"True Positive:  {TP}\n"
        + f"False Negative: {FN}\n"
        + f"False Positive: {FP}\n"
        + f"True Negative:  {TN}\n"
        + f"Precision: {P}\n"
        + f"Recall:    {R}"
    )
    logging.info(s)

    logging.info("Done!")


if __name__ == "__main__":
    main()
