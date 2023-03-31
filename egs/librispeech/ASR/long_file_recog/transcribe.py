#!/usr/bin/env python3
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
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

import argparse
import torch
import torch.nn as nn
import logging
from typing import List, Optional, Tuple

from pathlib import Path

import k2
import sentencepiece as spm
from asr_datamodule_trans import AsrDataModule
from beam_search import (
    fast_beam_search_one_best,
    greedy_search_batch,
    modified_beam_search,
)
from icefall.utils import AttributeDict, convert_timestamp
from lhotse import CutSet
from lhotse.supervision import AlignmentItem
from lhotse.serialization import SequentialJsonlWriter
from train import add_model_arguments, get_params, get_transducer_model


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--manifest-in-dir",
        type=Path,
        default=Path("data/manifests_chunk"),
        help="Path to directory with chunks cuts.",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/manifests_chunk_recog"),
        help="Path to directory to save the chunk cuts with recognition results.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=8,
        help="""Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It is a tuple. `len(value[0])` and `len(value[1])` are both
                 equal to the batch size. `value[0][i]` and `value[1][i]`
                 are the decoding result and timestamps for the i-th utterance
                 in the given batch respectively.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_LG, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return the decoding result, timestamps, and scores. See above description for the
      format of the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    if params.decoding_method == "fast_beam_search":
        res = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            return_timestamps=True,
        )
    elif params.decoding_method == "greedy_search":
        res = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            return_timestamps=True,
        )
    elif params.decoding_method == "modified_beam_search":
        res = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            return_timestamps=True,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    hyps = []
    timestamps = []
    scores = []
    for i in range(feature.shape[0]):
        hyps.append(res.hyps[i])
        timestamps.append(
            convert_timestamp(
                res.timestamps[i], params.subsampling_factor, params.frame_shift_ms
            )
        )
        scores.append(res.scores[i])

    return hyps, timestamps, scores


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    cuts_writer: SequentialJsonlWriter,
    decoding_graph: Optional[k2.Fsa] = None,
) -> None:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      cuts_writer:
        Writer to save the cuts with recognition results.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_LG, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains five elements:
      - cut_id
      - reference transcript
      - predicted result
      - timestamp of reference transcript
      - timestamp of predicted result
    """
    num_cuts = 0

    if params.decoding_method == "greedy_search":
        log_interval = 10
    else:
        log_interval = 10

    for batch_idx, batch in enumerate(dl):
        cuts = batch["supervisions"]["cut"]

        hyps, timestamps, scores = decode_one_batch(
            params=params,
            model=model,
            decoding_graph=decoding_graph,
            batch=batch,
        )

        for cut, symbol_list, time_list, score_list in zip(
            cuts, hyps, timestamps, scores
        ):
            ali = [
                AlignmentItem(symbol=symbol, start=start, duration=None, score=score)
                for symbol, start, score in zip(symbol_list, time_list, score_list)
            ]
            assert len(cut.supervisions) == 1, len(cut.supervisions)
            cut.supervisions[0].alignment = {"symbol": ali}
            cuts_writer.write(cut, flush=True)

        num_cuts += len(cuts)
        if batch_idx % log_interval == 0:
            logging.info(f"cuts processed until now is {num_cuts}")


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.frame_shift_ms = 10
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "fast_beam_search",
        "modified_beam_search",
    )

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    model.device = device

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    # we need cut ids to display recognition results.
    args.return_cuts = True
    asr_data_module = AsrDataModule(args)

    manifest_out_dir = params.manifest_out_dir
    manifest_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".jsonl.gz"
    subsets = ["librispeech_cuts_test-clean"]

    for subset in subsets:
        logging.info(f"Processing {subset}")

        out_cuts_filename = manifest_out_dir / (subset + suffix)
        if out_cuts_filename.is_file():
            logging.info(f"{out_cuts_filename} already exists - skipping.")
            continue

        in_cuts_filename = params.manifest_in_dir / (subset + suffix)
        cuts = asr_data_module.load_subset(in_cuts_filename)
        dl = asr_data_module.dataloaders(cuts)

        cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=False)
        decode_dataset(
            dl=dl,
            params=params,
            model=model,
            decoding_graph=decoding_graph,
            cuts_writer=cuts_writer,
        )
        cuts_writer.close()
        logging.info(f"Cuts saved to {out_cuts_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
