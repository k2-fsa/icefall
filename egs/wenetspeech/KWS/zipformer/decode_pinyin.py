#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao
#                                                 Mingshuang Luo)
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
(2) modified beam search
./zipformer/decode.py \
    --epoch 35 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --lang-dir data/lang_char \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4
"""


import argparse
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import WenetSpeechAsrDataModule
from beam_search import (
    keywords_search,
)
from lhotse.cut import Cut
from train import add_model_arguments, get_model, get_params

from icefall import ContextGraph
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
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
    text_to_pinyin,
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
        "--lang-dir",
        type=Path,
        default="data/lang_char",
        help="The lang dir containing word table and LG graph",
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
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
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

    parser.add_argument(
        "--pinyin-type",
        type=str,
        help="The type of pinyin used as the modeling units.",
    )

    parser.add_argument(
        "--keyword-file",
        type=str,
        help="File contains keywords.",
    )

    parser.add_argument(
        "--keyword-score",
        type=float,
        default=0.75,
        help="The threshold (probability) to boost the keyword.",
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    batch: dict,
    kws_graph: ContextGraph,
) -> Dict[str, List[List[str]]]:
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

    ans_dict = keywords_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        context_graph=kws_graph,
        beam=params.beam_size,
        num_tailing_blanks=8,
    )

    hyps = []
    for ans in ans_dict:
        hyp = []
        for hit in ans:
            hyp.append(
                (
                    hit.phrase,
                    (hit.timestamps[0], hit.timestamps[-1]),
                )
            )
        hyps.append(hyp)

    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    kws_graph: ContextGraph,
    keywords: Set[str],
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
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

    log_interval = 20

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
            lexicon=lexicon,
            kws_graph=kws_graph,
            batch=batch,
        )

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_words = list(ref_text)
            hyp_words = [x[0] for x in hyp_words]
            this_batch.append((cut_id, ref_words, list("".join(hyp_words))))
            hyp_set = set(hyp_words)
            hyp_str = " | ".join(hyp_words)
            for x in hyp_set:
                assert x in keywords, x
                if x in ref_text and x in keywords:
                    metric["all"].TP += 1
                    metric[x].TP += 1
                    metric[x].TP_list.append(f"({ref_text} -> {x})")
                if x not in ref_text and x in keywords:
                    metric["all"].FP += 1
                    metric[x].FP += 1
                    metric[x].FP_list.append(f"({ref_text} -> {x}/{cut_id})")
            for x in keywords:
                if x not in ref_text and x not in hyp_set:
                    metric["all"].TN += 1
                    metric[x].TN += 1

                if x in ref_text:
                    fn = True
                    for y in hyp_set:
                        if y in ref_text:
                            fn = False
                            break
                    if fn and ref_text.endswith(x):
                        metric["all"].FN += 1
                        metric[x].FN += 1
                        metric[x].FN_list.append(f"({ref_text} -> {hyp_str}/{cut_id})")

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

    print_s = ""
    with open(metric_filename, "w") as of:
        width = 10
        for key, item in sorted(
            metric.items(), key=lambda x: (x[1].FP, x[1].FN), reverse=True
        ):
            acc = (item.TP + item.TN) / (item.TP + item.TN + item.FP + item.FN)
            precision = (item.TP + 1) / (item.TP + item.FP + 1)
            recall = (item.TP + 1) / (item.TP + item.FN + 1)
            fpr = (item.FP + 1) / (item.FP + item.TN + 1)
            s = f"{key}:\n"
            s += f"\t{'TP':{width}}{'FP':{width}}{'FN':{width}}{'TN':{width}}\n"
            s += f"\t{str(item.TP):{width}}{str(item.FP):{width}}{str(item.FN):{width}}{str(item.TN):{width}}\n"
            s += f"\tAccuracy: {acc:.3f}\n"
            s += f"\tPrecision: {precision:.3f}\n"
            s += f"\tRecall(PPR): {recall:.3f}\n"
            s += f"\tFPR: {fpr:.3f}\n"
            s += f"\tF1: {2 * precision * recall / (precision + recall):.3f}\n"
            s += f"\tTP list: {' # '.join(item.TP_list)}\n"
            s += f"\tFP list: {' # '.join(item.FP_list)}\n"
            s += f"\tFN list: {' # '.join(item.FN_list)}\n"
            of.write(s + "\n")
            if key == "all":
                logging.info(s)

    logging.info("Wrote metric stats to {}".format(metric_filename))


@torch.no_grad()
def main():
    parser = get_parser()
    WenetSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "kws"

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

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"
    params.suffix += f"-keyword-score-{params.keyword_score}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    lexicon = Lexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    logging.info(params)

    keywords = []
    keywords_id = []
    with open(params.keyword_file, "r") as f:
        for line in f.readlines():
            score = 0
            kws = line.strip().upper().split()
            if kws[-1][0] == ":":
                score = float(kws[-1][1:])
                kws = kws[0:-1]
            tmp_ids = []
            kws = "".join(kws)
            kws_py = text_to_pinyin(kws, mode=params.pinyin_type)
            for k in kws_py:
                if k in lexicon.token_table:
                    tmp_ids.append(lexicon.token_table[k])
                else:
                    logging.warning(f"Containing OOV tokens, skipping line : {line}")
                    tmp_ids = []
                    break
            if tmp_ids:
                logging.info(f"Adding keyword : {kws}")
                keywords.append(kws)
                keywords_id.append((tmp_ids, score, kws))
    kws_graph = ContextGraph(context_score=params.keyword_score)
    kws_graph.build(keywords_id)

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
    wenetspeech = WenetSpeechAsrDataModule(args)

    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0

    def select_keywords(c: Cut):
        text = c.supervisions[0].text.strip()
        return text in keywords

    commands_cuts = wenetspeech.test_open_commands_cuts()
    commands_cuts = commands_cuts.filter(select_keywords)
    commands_cuts = commands_cuts.filter(remove_short_utt)
    commands_dl = wenetspeech.test_dataloaders(commands_cuts)

    test_net_cuts = wenetspeech.test_net_cuts()
    test_net_cuts = test_net_cuts.filter(remove_short_utt)
    test_net_dl = wenetspeech.test_dataloaders(test_net_cuts)

    test_sets = ["COMMANDS"]  # , "TEST_NET"]
    test_dls = [commands_dl]  # , test_net_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results, metric = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            lexicon=lexicon,
            kws_graph=kws_graph,
            keywords=set(keywords),
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
