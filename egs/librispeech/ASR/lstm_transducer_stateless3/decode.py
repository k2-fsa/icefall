#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
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
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./lstm_transducer_stateless2/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

To evaluate symbol delay, you should:
(1) Generate cuts with word-time alignments:
./local/add_alignment_librispeech.py \
    --alignments-dir data/alignment \
    --cuts-in-dir data/fbank \
    --cuts-out-dir data/fbank_ali
(2) Set the argument "--manifest-dir data/fbank_ali" while decoding.
For example:
./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 20 \
    --exp-dir ./lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method greedy_search \
    --manifest-dir data/fbank_ali
"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    DecodingResults,
    parse_hyp_and_timestamp,
    setup_logger,
    store_transcripts_and_timestamps,
    str2bool,
    write_error_stats_with_timestamps,
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
        default="lstm_transducer_stateless/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
        If you use fast_beam_search_nbest_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, Tuple[List[List[str]], List[List[float]]]]:
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
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return the decoding result and timestamps. See above description for the
      format of the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    # tail padding here to alleviate the tail deletion problem
    num_tail_padded_frames = 35
    feature = torch.nn.functional.pad(
        feature,
        (0, 0, 0, num_tail_padded_frames),
        mode="constant",
        value=LOG_EPS,
    )
    feature_lens += num_tail_padded_frames

    encoder_out, encoder_out_lens, _ = model.encoder(x=feature, x_lens=feature_lens)

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
    elif params.decoding_method == "fast_beam_search_nbest_LG":
        res = fast_beam_search_nbest_LG(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
            return_timestamps=True,
        )
    elif params.decoding_method == "fast_beam_search_nbest":
        res = fast_beam_search_nbest(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
            return_timestamps=True,
        )
    elif params.decoding_method == "fast_beam_search_nbest_oracle":
        res = fast_beam_search_nbest_oracle(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            ref_texts=sp.encode(supervisions["text"]),
            nbest_scale=params.nbest_scale,
            return_timestamps=True,
        )
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
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
        batch_size = encoder_out.size(0)
        tokens = []
        timestamps = []
        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i + 1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                res = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                    return_timestamps=True,
                )
            elif params.decoding_method == "beam_search":
                res = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                    return_timestamps=True,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            tokens.extend(res.tokens)
            timestamps.extend(res.timestamps)
        res = DecodingResults(hyps=tokens, timestamps=timestamps)

    hyps, timestamps = parse_hyp_and_timestamp(
        decoding_method=params.decoding_method,
        res=res,
        sp=sp,
        subsampling_factor=params.subsampling_factor,
        frame_shift_ms=params.frame_shift_ms,
        word_table=word_table,
    )

    if params.decoding_method == "greedy_search":
        return {"greedy_search": (hyps, timestamps)}
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
            if "LG" in params.decoding_method:
                key += f"_ngram_lm_scale_{params.ngram_lm_scale}"

        return {key: (hyps, timestamps)}
    else:
        return {f"beam_size_{params.beam_size}": (hyps, timestamps)}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str], List[float], List[float]]]]:
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
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
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

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        timestamps_ref = []
        for cut in batch["supervisions"]["cut"]:
            for s in cut.supervisions:
                time = []
                if s.alignment is not None and "word" in s.alignment:
                    time = [
                        aliword.start
                        for aliword in s.alignment["word"]
                        if aliword.symbol != ""
                    ]
                timestamps_ref.append(time)

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            word_table=word_table,
            batch=batch,
        )

        for name, (hyps, timestamps_hyp) in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts) and len(timestamps_hyp) == len(
                timestamps_ref
            )
            for cut_id, hyp_words, ref_text, time_hyp, time_ref in zip(
                cut_ids, hyps, texts, timestamps_hyp, timestamps_ref
            ):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words, time_ref, time_hyp))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[
        str,
        List[Tuple[List[str], List[str], List[str], List[float], List[float]]],
    ],
):
    test_set_wers = dict()
    test_set_delays = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts_and_timestamps(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer, mean_delay, var_delay = write_error_stats_with_timestamps(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer
            test_set_delays[key] = (mean_delay, var_delay)

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    test_set_delays = sorted(test_set_delays.items(), key=lambda x: x[1][0])
    delays_info = (
        params.res_dir
        / f"symbol-delay-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(delays_info, "w") as f:
        print("settings\tsymbol-delay", file=f)
        for key, val in test_set_delays:
            print(
                "{}\tmean: {}s, variance: {}".format(key, val[0], val[1]),
                file=f,
            )

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)

    s = "\nFor {}, symbol-delay of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_delays:
        s += "{}\tmean: {}s, variance: {}{}\n".format(key, val[0], val[1], note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_LG",
        "fast_beam_search_nbest_oracle",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
            if "LG" in params.decoding_method:
                params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

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

    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

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
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
