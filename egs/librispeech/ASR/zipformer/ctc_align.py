#!/usr/bin/env python3
#
# Copyright 2025  Brno University of Technology (Author: Karel Vesely)
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
Batch aligning with a CTC model (it can be Tranducer + CTC).
It works with both causal and non-causal models.
Streaming is disabled, or simulated by attention masks
(see: --chunk-size --left-context-frames).
Whole utterance processed by 1 forward() call.

Note: model averaging is present. With `--epoch 10 --avg 3`,
the epochs 8-10 are taken for averaging. Model averaging
is smoothing the CTC posteriors to some extent.

Usage:
(1) torchaudio forced_align()
./zipformer/ctc_align.py \
    --epoch 10 \
    --avg 3 \
    --exp-dir ./zipformer/exp \
    --max-duration 300 \
    --decoding-method ctc_align

"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Dict, List, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule as AsrDataModule
from lhotse import set_caching_enabled
from torchaudio.functional import (
    forced_align,
    merge_tokens,
)
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
    str2bool,
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
        "--res-dir-suffix",
        type=str,
        default="",
        help="Suffix to the directory, where alignments are stored.",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--ignored-tokens",
        type=str,
        nargs="+",
        default=[],
        help="List of BPE tokens to ignore when computing confidence scores "
        "(e.g., punctuation marks). Each token is a separate arg : "
        "`--ignore-tokens 'tok1' 'tok2' ...`",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="ctc_align",
        choices=[
            "ctc_align",
        ],
        help="Decoding method for doing the forced alignment.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "dataset_manifests",
        type=str,
        nargs="+",
        help="CutSet manifests to be aligned (CutSet with features and transcripts). "
        "Each CutSet as a separate arg : `manifest1 mainfest2 ...`",
    )

    add_model_arguments(parser)

    return parser


def align_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    ignored_tokens: set[int],
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Align one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for alignment.
               For now, just "ctc_alignment" is used.
        - value: It contains the alignment result: (labels, log_probs).
                 `len(value)` equals to batch size. `value[i]` is the alignment
                 result for the i-th utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      ignored_tokens:
        Set of int token-codes to be ignored for calculation of confidence.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.

    Returns:
      Return the alignment result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    batch_size = feature.shape[0]

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.causal:
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
    ctc_output = model.ctc_output(encoder_out)  # (N, T, C)

    hyps = []

    # tokenize the transcripts:
    text_encoded = sp.encode(supervisions["text"])

    # lengths
    num_tokens = [len(te) for te in text_encoded]
    max_tokens = max(num_tokens)

    # convert to padded np.array:
    targets = np.array(
        [
            np.pad(seq, (0, max_tokens - len(seq)), "constant", constant_values=-1)
            for seq in text_encoded
        ]
    )

    # convert to tensor:
    targets = torch.tensor(targets, dtype=torch.int32, device=device)
    target_lengths = torch.tensor(num_tokens, dtype=torch.int32, device=device)

    # torchaudio2.4.0+
    # The batch dimension for log_probs must be 1 at the current version:
    # https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/gpu/compute.cu#L277
    for ii in range(batch_size):
        labels, log_probs = forced_align(
            log_probs=ctc_output[ii:ii+1, : encoder_out_lens[ii]],
            targets=targets[ii, : target_lengths[ii]].unsqueeze(dim=0),
            input_lengths=encoder_out_lens[ii].unsqueeze(dim=0),
            target_lengths=target_lengths[ii].unsqueeze(dim=0),
            blank=params.blank_id,
        )

        # per-token time, score
        token_spans = merge_tokens(labels[0], log_probs[0].exp())
        # int -> token
        for s in token_spans:
            s.token = sp.id_to_piece(s.token)
        # mean conf. from the per-token scores
        mean_token_conf = np.mean([token_span.score for token_span in token_spans])

        # confidences
        ignore_mask = labels == 0
        for tok in ignored_tokens:
            ignore_mask += labels == tok

        nonblank_scores = log_probs[~ignore_mask].exp()
        num_scores = nonblank_scores.shape[0]

        if num_scores > 0:
            nonblank_min = float(nonblank_scores.min())
            nonblank_q05 = float(torch.quantile(nonblank_scores, 0.05))
            nonblank_q10 = float(torch.quantile(nonblank_scores, 0.10))
            nonblank_q20 = float(torch.quantile(nonblank_scores, 0.20))
            nonblank_q30 = float(torch.quantile(nonblank_scores, 0.30))
            mean_frame_conf = float(nonblank_scores.mean())
        else:
            nonblank_min = -1.0
            nonblank_q05 = -1.0
            nonblank_q10 = -1.0
            nonblank_q20 = -1.0
            nonblank_q30 = -1.0
            mean_frame_conf = -1.0

        if num_scores > 0:
            q0_20_conf = (nonblank_min + nonblank_q05 + nonblank_q10 + nonblank_q20) / 4
        else:
            q0_20_conf = 1.0  # default, no frames

        hyps.append(
            {
                "token_spans": token_spans,
                "mean_token_conf": mean_token_conf,
                "q0_20_conf": q0_20_conf,
                "num_scores": num_scores,
                "mean_frame_conf": mean_frame_conf,
                "nonblank_min": nonblank_min,
                "nonblank_q05": nonblank_q05,
                "nonblank_q10": nonblank_q10,
                "nonblank_q20": nonblank_q20,
                "nonblank_q30": nonblank_q30,
            }
        )

    return {"ctc_align": hyps}


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
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
    Returns:
      Return a dict, whose key is "ctc_align" (alignment method).
      Its value is a list of tuples. Each tuple is ternary, and it holds
      the a) utterance_key, b) reference transcript and c) dictionary
      with alignment results (token spans, confidences, etc).
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    ignored_tokens = set(params.ignored_tokens + ["<sos/eos>", "<unk>"])
    ignored_tokens_ints = [sp.piece_to_id(token) for token in ignored_tokens]

    logging.info(f"ignored tokens {ignored_tokens}")
    logging.info(f"ignored int codes {ignored_tokens_ints}")

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = align_one_batch(
            params=params,
            model=model,
            sp=sp,
            ignored_tokens=ignored_tokens_ints,
            batch=batch,
        )

        for name, alignments in hyps_dict.items():
            this_batch = []
            assert len(alignments) == len(texts)
            for cut_id, alignment, ref_text in zip(cut_ids, alignments, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, alignment))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        log_interval = 100
        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results


def save_alignment_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Save the token alignments and per-utterance confidences.
    """

    for key, results in results_dict.items():

        alignments_filename = params.res_dir / f"alignments-{test_set_name}.txt"

        time_step = 0.04

        with open(alignments_filename, "w", encoding="utf8") as fd:
            for key, ref_text, ali in results:
                for token_span in ali["token_spans"]:

                    t_beg = token_span.start * time_step
                    t_end = token_span.end * time_step
                    t_dur = t_end - t_beg
                    token = token_span.token
                    score = token_span.score

                    # CTM format : (wav_name, ch, t_beg, t_dur, token, score)
                    print(
                        f"{key} A {t_beg:.2f} {t_dur:.2f} {token} {score:.6f}", file=fd
                    )

        logging.info(f"The alignments are stored in `{alignments_filename}`")

        # ---------------------------

        confidences_filename = params.res_dir / f"confidences-{test_set_name}.txt"

        with open(confidences_filename, "w", encoding="utf8") as fd:
            print(
                "utterance_key mean_token_conf mean_frame_conf q0-20_conf "
                "(nonblank_min,q05,q10,q20,q30) (num_scores,num_tokens)",
                file=fd,
            )  # header

            for utterance_key, ref_text, ali in results:
                mean_token_conf = ali["mean_token_conf"]
                mean_frame_conf = ali["mean_frame_conf"]
                q0_20_conf = ali["q0_20_conf"]
                min_ = ali["nonblank_min"]
                q05 = ali["nonblank_q05"]
                q10 = ali["nonblank_q10"]
                q20 = ali["nonblank_q20"]
                q30 = ali["nonblank_q30"]

                num_scores = ali[
                    "num_scores"
                ]  # scores used to compute `mean_frame_conf`

                num_tokens = len(ali["token_spans"])  # tokens in ref transcript

                print(
                    f"{utterance_key} {mean_token_conf:.4f} {mean_frame_conf:.4f} "
                    f"{q0_20_conf:.4f} "
                    f"({min_:.4f},{q05:.4f},{q10:.4f},{q20:.4f},{q30:.4f}) "
                    f"({num_scores},{num_tokens})",
                    file=fd,
                )

        logging.info(f"The confidences are stored in `{confidences_filename}`")


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True)  # lhotse

    assert params.decoding_method in ("ctc_align",)
    assert params.enable_spec_aug is False
    assert params.use_ctc is True

    params.res_dir = params.exp_dir / (params.decoding_method + params.res_dir_suffix)

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}_avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}_avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"_chunk-{params.chunk_size}"
        params.suffix += f"_left-context-{params.left_context_frames}"

    params.suffix += f"_{params.decoding_method}"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-align-{params.suffix}")
    logging.info("Forced-alignment started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")  # unknown character, not an OOV
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
    asr_datamodule = AsrDataModule(args)

    # create array of dataloaders (one per test-set)
    testset_labels = []
    testset_dataloaders = []
    for testset_manifest in args.dataset_manifests:
        label = PurePath(testset_manifest).name  # basename
        label = label.replace(".jsonl.gz", "")

        test_cuts = asr_datamodule.load_manifest(testset_manifest)
        test_dataloader = asr_datamodule.test_dataloaders(test_cuts)

        testset_labels.append(label)
        testset_dataloaders.append(test_dataloader)

    # align
    for test_set, test_dl in zip(testset_labels, testset_dataloaders):
        results_dict = align_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_alignment_output(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
