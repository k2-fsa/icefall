#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Xiaoyu Yang,
#                                                 Wei Kang)
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
This script loads ONNX exported models and uses them to decode the test sets.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import MdccAsrDataModule
from lhotse.cut import Cut
from onnx_pretrained import OnnxModel, greedy_search

from icefall.utils import setup_logger, store_transcripts, write_error_stats


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner onnx model. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_char/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    return parser


def decode_one_batch(
    model: OnnxModel, token_table: k2.SymbolTable, batch: dict
) -> List[List[str]]:
    """Decode one batch and return the result.
    Currently it only greedy_search is supported.

    Args:
      model:
        The neural model.
      token_table:
        Mapping ids to tokens.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.

    Returns:
      Return the decoded results for each utterance.
    """
    feature = batch["inputs"]
    assert feature.ndim == 3
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(dtype=torch.int64)

    encoder_out, encoder_out_lens = model.run_encoder(x=feature, x_lens=feature_lens)

    hyps = greedy_search(
        model=model, encoder_out=encoder_out, encoder_out_lens=encoder_out_lens
    )

    hyps = [[token_table[h] for h in hyp] for hyp in hyps]
    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    token_table: k2.SymbolTable,
) -> Tuple[List[Tuple[str, List[str], List[str]]], float]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      model:
        The neural model.
      token_table:
        Mapping ids to tokens.

    Returns:
      - A list of tuples. Each tuple contains three elements:
         - cut_id,
         - reference transcript,
         - predicted result.
      - The total duration (in seconds) of the dataset.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    log_interval = 10
    total_duration = 0

    results = []
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        total_duration += sum([cut.duration for cut in batch["supervisions"]["cut"]])

        hyps = decode_one_batch(model=model, token_table=token_table, batch=batch)

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_words = list(ref_text)
            this_batch.append((cut_id, ref_words, hyp_words))

        results.extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results, total_duration


def save_results(
    res_dir: Path,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
):
    recog_path = res_dir / f"recogs-{test_set_name}.txt"
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = res_dir / f"errs-{test_set_name}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    errs_info = res_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("WER", file=f)
        print(wer, file=f)

    s = "\nFor {}, WER is {}:\n".format(test_set_name, wer)
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    MdccAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    assert (
        args.decoding_method == "greedy_search"
    ), "Only supports greedy_search currently."
    res_dir = Path(args.exp_dir) / f"onnx-{args.decoding_method}"

    setup_logger(f"{res_dir}/log-decode")
    logging.info("Decoding started")

    device = torch.device("cpu")
    logging.info(f"Device: {device}")

    token_table = k2.SymbolTable.from_file(args.tokens)
    assert token_table[0] == "<blk>"

    logging.info(vars(args))

    logging.info("About to create model")
    model = OnnxModel(
        encoder_model_filename=args.encoder_model_filename,
        decoder_model_filename=args.decoder_model_filename,
        joiner_model_filename=args.joiner_model_filename,
    )

    # we need cut ids to display recognition results.
    args.return_cuts = True

    mdcc = MdccAsrDataModule(args)

    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0

    valid_cuts = mdcc.valid_cuts()
    valid_cuts = valid_cuts.filter(remove_short_utt)
    valid_dl = mdcc.valid_dataloaders(valid_cuts)

    test_cuts = mdcc.test_net_cuts()
    test_cuts = test_cuts.filter(remove_short_utt)
    test_dl = mdcc.test_dataloaders(test_cuts)

    test_sets = ["valid", "test"]
    test_dl = [valid_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        start_time = time.time()
        results, total_duration = decode_dataset(
            dl=test_dl, model=model, token_table=token_table
        )
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        rtf = elapsed_seconds / total_duration

        logging.info(f"Elapsed time: {elapsed_seconds:.3f} s")
        logging.info(f"Wave duration: {total_duration:.3f} s")
        logging.info(
            f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
        )

        save_results(res_dir=res_dir, test_set_name=test_set, results=results)

    logging.info("Done!")


if __name__ == "__main__":
    main()
