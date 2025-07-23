#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
(1) greedy search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
"""


import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule

from lhotse import set_caching_enabled
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


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
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=8,
        help="""The number of time-steps in denoising decoding."""
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-04,
        help="""The t value that we start from with pure noise."""
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--skip-scoring",
        type=str2bool,
        default=False,
        help="""Skip scoring, but still save the ASR output (for eval sets).""",
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> List[List[str]]:
    """Decode one batch and return the result as a list of sentences
       (each sentence is a list of words).

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        The return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result as a list of list of strings (words), i.e.
      a list of sentences.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    tokens = model.infer(feature, feature_lens, params.eps, params.num_steps)  # list of lists of int

    hyps = [ sp.decode(t).split() for t in tokens ]  # list of lists of str

    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> List[Tuple[str, List[str], List[str]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Returns list of tuples (cut_id, ref_transcript, hyp_transcript)
      with types (str, List[str], List[str]).
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"


    log_interval = 10


    results = [ ]
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_words = ref_text.split()
            this_batch.append((cut_id, ref_words, hyp_words))
        results.extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]]
):
    """
    Save text produced by ASR.
    """
    recogs_filename = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"

    results = sorted(results)
    store_transcripts(filename=recogs_filename, texts=results)

    logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str], Tuple]],
):
    """
    Save WER and per-utterance word alignments.
    """


    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
    with open(errs_filename, "w", encoding="utf8") as fd:
        wer = write_error_stats(
            fd, f"{test_set_name}", results, enable_log=True
        )
    logging.info(f"Wrote detailed error stats to {errs_filename}")

    wer_filename = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"

    with open(wer_filename, "w", encoding="utf8") as fd:
        print(f"{wer}", file=fd)

    s = f"\nFor {test_set_name}, WER is {wer}"
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True)  # lhotse

    params.res_dir = params.exp_dir / "decode"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}_avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}_avg-{params.avg}"

    params.suffix += f"_{params.num_steps}step"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
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
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()
    dev_clean_cuts = librispeech.dev_clean_cuts()
    dev_other_cuts = librispeech.dev_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)
    dev_clean_dl = librispeech.test_dataloaders(dev_clean_cuts)
    dev_other_dl = librispeech.test_dataloaders(dev_other_cuts)

    test_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    test_dl = [dev_clean_dl, dev_other_dl, test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_asr_output(
            params=params,
            test_set_name=test_set,
            results=results,
        )

        if not params.skip_scoring:
            save_wer_results(
                params=params,
                test_set_name=test_set,
                results=results,
            )

    logging.info("Done!")


if __name__ == "__main__":
    main()
