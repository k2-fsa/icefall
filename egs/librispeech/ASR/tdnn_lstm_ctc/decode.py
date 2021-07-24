#!/usr/bin/env python3


import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import k2
import torch
import torch.nn as nn
from model import TdnnLstm

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.dataset.librispeech import LibriSpeechAsrDataModule
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=9,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "exp_dir": Path("tdnn_lstm_ctc/exp3/"),
            "lang_dir": Path("data/lang"),
            "feature_dim": 80,
            "subsampling_factor": 3,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
        }
    )
    return params


@torch.no_grad()
def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    HLG: k2.Fsa,
    batch: dict,
    lexicon: Lexicon,
) -> List[Tuple[List[str], List[str]]]:
    """Decode one batch and return a list of tuples containing
    `(ref_words, hyp_words)`.

    Args:
      params:
        It is the return value of :func:`get_params`.


    """
    device = HLG.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is [N, T, C]

    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]

    nnet_output = model(feature)
    # nnet_output is [N, T, C]

    supervisions = batch["supervisions"]

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"] // params.subsampling_factor,
            supervisions["num_frames"] // params.subsampling_factor,
        ),
        1,
    ).to(torch.int32)

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    lattices = k2.intersect_dense_pruned(
        HLG,
        dense_fsa_vec,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
    )

    best_paths = k2.shortest_path(lattices, use_double_scores=True)

    hyps = get_texts(best_paths)
    hyps = [[lexicon.words[i] for i in ids] for ids in hyps]

    texts = supervisions["text"]

    results = []
    for hyp_words, ref_text in zip(hyps, texts):
        ref_words = ref_text.split()
        results.append((ref_words, hyp_words))
    return results


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_phone_id = max(lexicon.tokens)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    HLG = k2.Fsa.from_dict(torch.load("data/lm/HLG.pt"))
    HLG = HLG.to(device)
    assert HLG.requires_grad is False

    model = TdnnLstm(
        num_features=params.feature_dim,
        num_classes=max_phone_id + 1,  # +1 for the blank symbol
        subsampling_factor=params.subsampling_factor,
    )
    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.load_state_dict(average_checkpoints(filenames))

    model.to(device)
    model.eval()

    librispeech = LibriSpeechAsrDataModule(args)
    # CAUTION: `test_sets` is for displaying only.
    # If you want to skip test-clean, you have to skip
    # it inside the for loop. That is, use
    #
    #   if test_set == 'test-clean': continue
    #
    test_sets = ["test-clean", "test-other"]
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        tot_num_cuts = len(test_dl.dataset.cuts)
        num_cuts = 0

        results = []
        for batch_idx, batch in enumerate(test_dl):
            this_batch = decode_one_batch(
                params=params,
                model=model,
                HLG=HLG,
                batch=batch,
                lexicon=lexicon,
            )
            results.extend(this_batch)

            num_cuts += len(batch["supervisions"]["text"])

            if batch_idx % 100 == 0:
                logging.info(
                    f"batch {batch_idx}, cuts processed until now is "
                    f"{num_cuts}/{tot_num_cuts} "
                    f"({float(num_cuts)/tot_num_cuts*100:.6f}%)"
                )

        errs_filename = params.exp_dir / f"errs-{test_set}.txt"
        with open(errs_filename, "w") as f:
            write_error_stats(f, test_set, results)

    logging.info("Done!")


if __name__ == "__main__":
    main()
