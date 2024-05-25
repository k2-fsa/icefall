#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
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
Usage:
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --keywords-file keywords.txt \
    --beam-size 4
"""

import argparse
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import GigaSpeechAsrDataModule
from beam_search import keywords_search
from lhotse.cut import Cut
from train import add_model_arguments, get_model, get_params

from icefall import ContextGraph
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


@dataclass
class KwMetric:
    TP: int = 0  # True positive
    FN: int = 0  # False negative
    FP: int = 0  # False positive
    TN: int = 0  # True negative
    FN_list: List[str] = field(default_factory=list)
    FP_list: List[str] = field(default_factory=list)
    TP_list: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"(TP:{self.TP}, FN:{self.FN}, FP:{self.FP}, TN:{self.TN})"


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
        "--beam",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--keywords-file",
        type=str,
        help="File contains keywords.",
    )

    parser.add_argument(
        "--test-set",
        type=str,
        default="small",
        help="small or large",
    )

    parser.add_argument(
        "--keywords-score",
        type=float,
        default=1.5,
        help="""
        The default boosting score (token level) for keywords. it will boost the
        paths that match keywords to make them survive beam search.
        """,
    )

    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.35,
        help="The default threshold (probability) to trigger the keyword.",
    )

    parser.add_argument(
        "--num-tailing-blanks",
        type=int,
        default=1,
        help="The number of tailing blanks should have after hitting one keyword.",
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
    sp: spm.SentencePieceProcessor,
    batch: dict,
    keywords_graph: Optional[ContextGraph] = None,
) -> List[List[Tuple[str, Tuple[int, int]]]]:
    """Decode one batch and return the result in a list.

    The length of the list equals to batch size, the i-th element contains the
    triggered keywords for the i-th utterance in the given batch. The triggered
    keywords are also a list, each of it contains a tuple of hitting keyword and
    the corresponding start timestamps and end timestamps of the hitting keyword.

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
      keywords_graph:
        The graph containing keywords.
    Returns:
      Return the decoding result. See above description for the format of
      the returned list.
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

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    ans_dict = keywords_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        keywords_graph=keywords_graph,
        beam=params.beam,
        num_tailing_blanks=params.num_tailing_blanks,
        blank_penalty=params.blank_penalty,
    )

    hyps = []
    for ans in ans_dict:
        hyp = []
        for hit in ans:
            hyp.append((hit.phrase, (hit.timestamps[0], hit.timestamps[-1])))
        hyps.append(hyp)

    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    keywords_graph: ContextGraph,
    keywords: Set[str],
    test_only_keywords: bool,
) -> Tuple[List[Tuple[str, List[str], List[str]]], KwMetric]:
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
      keywords_graph:
        The graph containing keywords.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    log_interval = 50

    results = []
    metric = {"all": KwMetric()}
    for k in keywords:
        metric[k] = KwMetric()

    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            keywords_graph=keywords_graph,
            batch=batch,
        )

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_text = ref_text.upper()
            ref_words = ref_text.split()
            hyp_words = [x[0] for x in hyp_words]
            # for computing WER
            this_batch.append((cut_id, ref_words, " ".join(hyp_words).split()))
            hyp_set = set(hyp_words)  # each item is a keyword phrase
            if len(hyp_words) > 1:
                logging.warning(
                    f"Cut {cut_id} triggers more than one keywords : {hyp_words},"
                    f"please check the transcript to see if it really has more "
                    f"than one keywords, if so consider splitting this audio and"
                    f"keep only one keyword for each audio."
                )
            hyp_str = " | ".join(
                hyp_words
            )  # The triggered keywords for this utterance.
            TP = False
            FP = False
            for x in hyp_set:
                assert x in keywords, x  # can only trigger keywords
                if (test_only_keywords and x == ref_text) or (
                    not test_only_keywords and x in ref_text
                ):
                    TP = True
                    metric[x].TP += 1
                    metric[x].TP_list.append(f"({ref_text} -> {x})")
                if (test_only_keywords and x != ref_text) or (
                    not test_only_keywords and x not in ref_text
                ):
                    FP = True
                    metric[x].FP += 1
                    metric[x].FP_list.append(f"({ref_text} -> {x})")
            if TP:
                metric["all"].TP += 1
            if FP:
                metric["all"].FP += 1
            TN = True  # all keywords are true negative then the summery is true negative.
            FN = False
            for x in keywords:
                if x not in ref_text and x not in hyp_set:
                    metric[x].TN += 1
                    continue

                TN = False
                if (test_only_keywords and x == ref_text) or (
                    not test_only_keywords and x in ref_text
                ):
                    fn = True
                    for y in hyp_set:
                        if (test_only_keywords and y == ref_text) or (
                            not test_only_keywords and y in ref_text
                        ):
                            fn = False
                            break
                    if fn:
                        FN = True
                        metric[x].FN += 1
                        metric[x].FN_list.append(f"({ref_text} -> {hyp_str})")
            if TN:
                metric["all"].TN += 1
            if FN:
                metric["all"].FN += 1

        results.extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results, metric


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
    metric: KwMetric,
):
    recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)
    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    metric_filename = params.res_dir / f"metric-{test_set_name}-{params.suffix}.txt"

    with open(metric_filename, "w") as of:
        width = 10
        for key, item in sorted(
            metric.items(), key=lambda x: (x[1].FP, x[1].FN), reverse=True
        ):
            acc = (item.TP + item.TN) / (item.TP + item.TN + item.FP + item.FN)
            precision = (
                0.0 if (item.TP + item.FP) == 0 else item.TP / (item.TP + item.FP)
            )
            recall = 0.0 if (item.TP + item.FN) == 0 else item.TP / (item.TP + item.FN)
            fpr = 0.0 if (item.FP + item.TN) == 0 else item.FP / (item.FP + item.TN)
            s = f"{key}:\n"
            s += f"\t{'TP':{width}}{'FP':{width}}{'FN':{width}}{'TN':{width}}\n"
            s += f"\t{str(item.TP):{width}}{str(item.FP):{width}}{str(item.FN):{width}}{str(item.TN):{width}}\n"
            s += f"\tAccuracy: {acc:.3f}\n"
            s += f"\tPrecision: {precision:.3f}\n"
            s += f"\tRecall(PPR): {recall:.3f}\n"
            s += f"\tFPR: {fpr:.3f}\n"
            s += f"\tF1: {0.0 if precision * recall == 0 else 2 * precision * recall / (precision + recall):.3f}\n"
            if key != "all":
                s += f"\tTP list: {' # '.join(item.TP_list)}\n"
                s += f"\tFP list: {' # '.join(item.FP_list)}\n"
                s += f"\tFN list: {' # '.join(item.FN_list)}\n"
            of.write(s + "\n")
            if key == "all":
                logging.info(s)
        of.write(f"\n\n{params.keywords_config}")

    logging.info("Wrote metric stats to {}".format(metric_filename))


@torch.no_grad()
def main():
    parser = get_parser()
    GigaSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "kws"

    params.suffix = params.test_set
    if params.iter > 0:
        params.suffix += f"-iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix += f"-epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"-chunk-{params.chunk_size}"
        params.suffix += f"-left-context-{params.left_context_frames}"

    params.suffix += f"-score-{params.keywords_score}"
    params.suffix += f"-threshold-{params.keywords_threshold}"
    params.suffix += f"-tailing-blanks-{params.num_tailing_blanks}"
    if params.blank_penalty != 0:
        params.suffix += f"-blank-penalty-{params.blank_penalty}"
    params.suffix += f"-keywords-{params.keywords_file.split('/')[-1]}"

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

    phrases = []
    token_ids = []
    keywords_scores = []
    keywords_thresholds = []
    keywords_config = []
    with open(params.keywords_file, "r") as f:
        for line in f.readlines():
            keywords_config.append(line)
            score = 0
            threshold = 0
            keyword = []
            words = line.strip().upper().split()
            for word in words:
                word = word.strip()
                if word[0] == ":":
                    score = float(word[1:])
                    continue
                if word[0] == "#":
                    threshold = float(word[1:])
                    continue
                keyword.append(word)
            keyword = " ".join(keyword)
            phrases.append(keyword)
            token_ids.append(sp.encode(keyword))
            keywords_scores.append(score)
            keywords_thresholds.append(threshold)

    params.keywords_config = "".join(keywords_config)

    keywords_graph = ContextGraph(
        context_score=params.keywords_score, ac_threshold=params.keywords_threshold
    )
    keywords_graph.build(
        token_ids=token_ids,
        phrases=phrases,
        scores=keywords_scores,
        ac_thresholds=keywords_thresholds,
    )
    keywords = set(phrases)

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
    gigaspeech = GigaSpeechAsrDataModule(args)

    test_cuts = gigaspeech.test_cuts()
    test_dl = gigaspeech.test_dataloaders(test_cuts)

    if params.test_set == "small":
        test_fsc_small_cuts = gigaspeech.fsc_test_small_cuts()
        test_fsc_small_dl = gigaspeech.test_dataloaders(test_fsc_small_cuts)
        test_sets = ["small-fsc", "test"]
        test_dls = [test_fsc_small_dl, test_dl]
    else:
        assert params.test_set == "large", params.test_set
        test_fsc_large_cuts = gigaspeech.fsc_test_large_cuts()
        test_fsc_large_dl = gigaspeech.test_dataloaders(test_fsc_large_cuts)
        test_sets = ["large-fsc", "test"]
        test_dls = [test_fsc_large_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results, metric = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            keywords_graph=keywords_graph,
            keywords=keywords,
            test_only_keywords="fsc" in test_set,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results=results,
            metric=metric,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
