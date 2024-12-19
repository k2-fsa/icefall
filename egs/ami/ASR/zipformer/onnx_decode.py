#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Xiaoyu Yang)
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

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model to ONNX

./zipformer/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --causal False

It will generate the following 3 files inside $repo/exp:

  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx

2. Run this file

./zipformer/onnx_decode.py \
  --exp-dir $repo/exp \
  --max-duration 600 \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
"""


import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule

from onnx_pretrained import greedy_search, OnnxModel

from icefall.utils import setup_logger, store_transcripts, write_error_stats
from k2 import SymbolTable


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
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    return parser


def decode_one_batch(
    model: OnnxModel, token_table: SymbolTable, batch: dict
) -> List[List[str]]:
    """Decode one batch and return the result.
    Currently it only greedy_search is supported.

    Args:
      model:
        The neural model.
      token_table:
        The token table.
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

    def token_ids_to_words(token_ids: List[int]) -> str:
        text = ""
        for i in token_ids:
            text += token_table[i]
        return text.replace("▁", " ").strip()

    hyps = [token_ids_to_words(h).split() for h in hyps]
    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    token_table: SymbolTable,
) -> Tuple[List[Tuple[str, List[str], List[str]]], float]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      model:
        The neural model.
      token_table:
        The token table.

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
            ref_words = ref_text.split()
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
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    assert (
        args.decoding_method == "greedy_search"
    ), "Only supports greedy_search currently."
    res_dir = Path(args.exp_dir) / f"onnx-{args.decoding_method}"

    setup_logger(f"{res_dir}/log-decode")
    logging.info("Decoding started")

    device = torch.device("cpu")
    logging.info(f"Device: {device}")

    token_table = SymbolTable.from_file(args.tokens)

    logging.info(vars(args))

    logging.info("About to create model")
    model = OnnxModel(
        encoder_model_filename=args.encoder_model_filename,
        decoder_model_filename=args.decoder_model_filename,
        joiner_model_filename=args.joiner_model_filename,
    )

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

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
