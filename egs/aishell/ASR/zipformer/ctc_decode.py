#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Liyong Guo,
#                                                 Quandong Wang,
#                                                 Zengwei Yao,
#                                                 Zhifeng Han,)
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

(1) ctc-greedy-search (with cr-ctc)
./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 24 \
    --exp-dir ./zipformer/exp \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --use-transducer 0 \
    --max-duration 600 \
    --decoding-method ctc-greedy-search
"""


import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule
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
from lhotse.cut import Cut
from train import add_model_arguments, get_model, get_params

from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.context_graph import ContextGraph, ContextState
from icefall.decode import (
    ctc_greedy_search,
    ctc_prefix_beam_search,
    ctc_prefix_beam_search_attention_decoder_rescoring,
    ctc_prefix_beam_search_shallow_fussion,
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder_no_ngram,
    rescore_with_attention_decoder_with_ngram,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    DecodingResults,
    make_pad_mask,
    setup_logger,
    store_transcripts_and_timestamps_withoutref,
    str2bool,
    write_error_stats,
    parse_hyp_and_timestamp_ch,
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
        "--lang-dir",
        type=Path,
        default="data/lang_char",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="ctc-greedy-search",
        help="""Decoding method.
        Supported values are:
        - (1) ctc-greedy-search. Use CTC greedy search. It uses a sentence piece
          model, i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
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
        fast_beam_search, fast_beam_search_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--ilme-scale",
        type=float,
        default=0.2,
        help="""
        Used only when --decoding_method is fast_beam_search_LG.
        It specifies the scale for the internal language model estimation.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search, fast_beam_search_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search, fast_beam_search_LG,
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
        Used only when the decoding method is fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    add_model_arguments(parser)

    return parser

def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    graph_compiler: CharCtcTrainingGraphCompiler,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, Tuple[List[List[str]], List[List[Tuple[float, float]]]]]:
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
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
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
    feature_lens = supervisions["num_frames"].to(device)

    if params.causal:
        # this seems to cause insertions at the end of the utterance if used with zipformer.
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    x, x_lens = model.encoder_embed(feature, feature_lens)

    src_key_padding_mask = make_pad_mask(x_lens)
    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

    encoder_out, encoder_out_lens = model.encoder(x, x_lens, src_key_padding_mask)
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

    ctc_output = model.ctc_output(encoder_out)  # (N, T, C)

    hyps = []

    if params.decoding_method == "ctc-greedy-search" and params.max_sym_per_frame == 1:
        res = ctc_greedy_search(
            ctc_output=ctc_output,
            encoder_out_lens=encoder_out_lens,
            return_timestamps = True,
        )
    else:
        raise ValueError(
            f"Unsupported decoding method: {params.decoding_method}"
        )
    
    hyps, timestamps = parse_hyp_and_timestamp_ch(
        res=res,
        subsampling_factor=params.subsampling_factor,
        word_table = lexicon.token_table,
        # frame_shift_ms=params.frame_shift_ms,
    )

    key = f"blank_penalty_{params.blank_penalty}"
    if params.decoding_method == "ctc-greedy-search":
        return {"ctc-greedy-search_" + key: (hyps, timestamps)}
    elif "fast_beam_search" in params.decoding_method:
        key += f"_beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
        if "LG" in params.decoding_method:
            key += f"_ilme_scale_{params.ilme_scale}"
            key += f"_ngram_lm_scale_{params.ngram_lm_scale}"

        return {key: (hyps, timestamps)}
    else:
        return {f"beam_size_{params.beam_size}_" + key: (hyps, timestamps)}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    graph_compiler: CharCtcTrainingGraphCompiler,
    decoding_graph: Optional[k2.Fsa] = None,
    with_timestamp: bool = False,
) -> Dict[str, List[Tuple[str, List[str], List[str], List[Tuple[float, float]]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      with_timestamp:
        Whether to decode with timestamp.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains 4 elements:
      Respectively, they are cut_id, the reference transcript, the predicted result and the decoded_timestamps.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "ctc-greedy-search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        texts = [list("".join(text.split())) for text in texts]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            lexicon=lexicon,
            graph_compiler=graph_compiler,
            decoding_graph=decoding_graph,
            batch=batch,
        )
        if with_timestamp:

            for name, (hyps, timestamps_hyp) in hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(texts) and len(timestamps_hyp) == len(hyps)
                for cut_id, hyp_words, ref_text, time_hyp in zip(
                    cut_ids, hyps, texts, timestamps_hyp
                ):
                    this_batch.append((cut_id, ref_text, hyp_words, time_hyp))

                results[name].extend(this_batch)
        else:

            for name, hyps in hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(texts)
                for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                    this_batch.append((cut_id, ref_text, hyp_words))

                results[name].extend(this_batch)
                
        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str], List[Tuple[float, float]]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts_and_timestamps_withoutref(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        result_without_timestamp = [(res[0], res[1], res[2]) for res in results]
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f,
                f"{test_set_name}-{key}",
                result_without_timestamp,
                enable_log=True,
                compute_CER=True,
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    # add decoding params
    # params.update(get_decoding_params())
    params.update(vars(args))

    assert params.decoding_method in (
        "ctc-greedy-search",
    ) # only support ctc-greedy-search
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"-chunk-{params.chunk_size}"
        params.suffix += f"-left-context-{params.left_context_frames}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
        if "LG" in params.decoding_method:
            params.suffix += f"_ilme_scale_{params.ilme_scale}"
            params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"
    params.suffix += f"-blank-penalty-{params.blank_penalty}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    params.device = device

    logging.info(f"Device: {device}")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )

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

    if "fast_beam_search" in params.decoding_method:
        if "LG" in params.decoding_method:
            lexicon = Lexicon(params.lang_dir)
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    aishell = AishellAsrDataModule(args)

    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0

    dev_cuts = aishell.valid_cuts()
    dev_cuts = dev_cuts.filter(remove_short_utt)
    dev_dl = aishell.valid_dataloaders(dev_cuts)

    test_cuts = aishell.test_cuts()
    test_cuts = test_cuts.filter(remove_short_utt)
    test_dl = aishell.test_dataloaders(test_cuts)

    test_sets = ["dev", "test"]
    test_dls = [dev_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            lexicon=lexicon,
            graph_compiler=graph_compiler,
            decoding_graph=decoding_graph,
            with_timestamp=True,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
