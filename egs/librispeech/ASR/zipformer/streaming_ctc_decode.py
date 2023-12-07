#!/usr/bin/env python3
# Copyright 2022-2023 Xiaomi Corporation (Authors: Wei Kang,
#                                                  Fangjun Kuang,
#                                                  Zengwei Yao,
#                                                  Zengrui Jin,)
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
./zipformer/streaming_decode.py \
  --epoch 28 \
  --avg 15 \
  --causal 1 \
  --chunk-size 32 \
  --left-context-frames 256 \
  --exp-dir ./zipformer/exp \
  --decoding-method greedy_search \
  --num-decode-streams 2000
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from ctc_decode_stream import DecodeStream
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet
from streaming_decode import (
    get_init_states,
    stack_states,
    streaming_forward,
    unstack_states,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import get_lattice, one_best_decoding
from icefall.utils import (
    AttributeDict,
    get_texts,
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
        default=20,
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
        default=1,
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
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Supported decoding methods are:
        greedy_search
        modified_beam_search
        fast_beam_search
        """,
    )

    parser.add_argument(
        "--num_active_paths",
        type=int,
        default=4,
        help="""An interger indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded parallel.",
    )

    add_model_arguments(parser)

    return parser


def get_decoding_params() -> AttributeDict:
    """Parameters for decoding."""
    params = AttributeDict(
        {
            "feature_dim": 80,
            "subsampling_factor": 4,
            "frame_shift_ms": 10,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def decode_one_chunk(
    params: AttributeDict,
    model: nn.Module,
    H: Optional[k2.Fsa],
    intersector: k2.OnlineDenseIntersecter,
    decode_streams: List[DecodeStream],
    streams_to_pad: int = None,
) -> List[int]:
    """Decode one chunk frames of features for each decode_streams and
    return the indexes of finished streams in a List.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      decode_streams:
        A List of DecodeStream, each belonging to a utterance.
    Returns:
      Return a List containing which DecodeStreams are finished.
    """
    device = model.device
    chunk_size = int(params.chunk_size)
    left_context_len = int(params.left_context_frames)

    features = []
    feature_lens = []
    states = []
    processed_lens = []  # Used in fast-beam-search

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(chunk_size * 2)
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    # Make sure the length after encoder_embed is at least 1.
    # The encoder_embed subsample features (T - 7) // 2
    # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
    tail_length = chunk_size * 2 + 7 + 2 * 3
    if features.size(1) < tail_length:
        pad_length = tail_length - features.size(1)
        feature_lens += pad_length
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, pad_length),
            mode="constant",
            value=LOG_EPS,
        )

    states = stack_states(states)

    encoder_out, encoder_out_lens, new_states = streaming_forward(
        features=features,
        feature_lens=feature_lens,
        model=model,
        states=states,
        chunk_size=chunk_size,
        left_context_len=left_context_len,
    )
    ctc_output = model.ctc_output(encoder_out)  # (N, T, C)

    if streams_to_pad:
        ctc_output = torch.cat(
            [
                ctc_output,
                torch.zeros(
                    (streams_to_pad, ctc_output.size(-2), ctc_output.size(-1)),
                    device=device,
                ),
            ]
        )

    supervision_segments = torch.tensor(
        [[i, 0, 8] for i in range(params.num_decode_streams)],
        dtype=torch.int32,
    )

    # decoding_graph = H

    # lattice = get_lattice(
    #     nnet_output=ctc_output,
    #     decoding_graph=decoding_graph,
    #     supervision_segments=supervision_segments,
    #     search_beam=params.search_beam,
    #     output_beam=params.output_beam,
    #     min_active_states=params.min_active_states,
    #     max_active_states=params.max_active_states,
    #     subsampling_factor=params.subsampling_factor,
    # )
    dense_fsa_vec = k2.DenseFsaVec(ctc_output, supervision_segments)

    current_decode_states = [
        decode_stream.decode_state for decode_stream in decode_streams
    ]
    if streams_to_pad:
        current_decode_states += [k2.DecodeStateInfo()] * streams_to_pad
    lattice, current_decode_states = intersector.decode(
        dense_fsa_vec, current_decode_states
    )

    best_path = one_best_decoding(
        lattice=lattice, use_double_scores=params.use_double_scores
    )

    # Note: `best_path.aux_labels` contains token IDs, not word IDs
    # since we are using H, not HLG here.
    #
    # token_ids is a lit-of-list of IDs
    token_ids = get_texts(best_path)

    states = unstack_states(new_states)

    num_streams = (
        len(decode_streams) - streams_to_pad if streams_to_pad else len(decode_streams)
    )

    finished_streams = []
    for i in range(num_streams):
        decode_streams[i].hyp += token_ids[i]
        decode_streams[i].states = states[i]
        decode_streams[i].decode_state = current_decode_states[i]
        decode_streams[i].done_frames += encoder_out_lens[i]
        if decode_streams[i].done:
            finished_streams.append(i)

    return finished_streams


def decode_dataset(
    cuts: CutSet,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      cuts:
        Lhotse Cutset containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    device = model.device

    opts = FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80

    log_interval = 100

    intersector = k2.OnlineDenseIntersecter(
        decoding_graph=decoding_graph,
        num_streams=params.num_decode_streams,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
    )

    decode_results = []
    # Contain decode streams currently running.
    decode_streams = []
    for num, cut in enumerate(cuts):
        # each utterance has a DecodeStream.
        initial_states = get_init_states(model=model, batch_size=1, device=device)
        decode_stream = DecodeStream(
            params=params,
            cut_id=cut.id,
            initial_states=initial_states,
            decode_state=k2.DecodeStateInfo(),
            device=device,
        )

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"

        samples = torch.from_numpy(audio).squeeze(0)

        fbank = Fbank(opts)
        feature = fbank(samples.to(device))
        decode_stream.set_features(feature, tail_pad_len=30)
        decode_stream.ground_truth = cut.supervisions[0].text

        decode_streams.append(decode_stream)

        while len(decode_streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                params=params,
                model=model,
                H=decoding_graph,
                intersector=intersector,
                decode_streams=decode_streams,
            )
            for i in sorted(finished_streams, reverse=True):
                decode_results.append(
                    (
                        decode_streams[i].id,
                        decode_streams[i].ground_truth.split(),
                        sp.decode(decode_streams[i].decoding_result()).split(),
                    )
                )
                del decode_streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    num_remained_decode_streams = len(decode_streams)
    # decode final chunks of last sequences
    while num_remained_decode_streams:
        finished_streams = decode_one_chunk(
            params=params,
            model=model,
            H=decoding_graph,
            intersector=intersector,
            decode_streams=decode_streams,
            streams_to_pad=params.num_decode_streams - num_remained_decode_streams,
        )
        for i in sorted(finished_streams, reverse=True):
            decode_results.append(
                (
                    decode_streams[i].id,
                    decode_streams[i].ground_truth.split(),
                    sp.decode(decode_streams[i].decoding_result()).split(),
                )
            )
            del decode_streams[i]
            num_remained_decode_streams -= 1

    key = "ctc-decoding"
    return {key: decode_results}


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
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
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params = get_decoding_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    assert params.causal, params.causal
    assert "," not in params.chunk_size, "chunk_size should be one value in decoding."
    assert (
        "," not in params.left_context_frames
    ), "left_context_frames should be one value in decoding."
    params.suffix += f"-chunk-{params.chunk_size}"
    params.suffix += f"-left-context-{params.left_context_frames}"

    # for fast_beam_search
    if params.decoding_method == "fast_beam_search":
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"

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

    # <blk> and <unk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()
    max_token_id = sp.get_piece_size() - 1

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
                if start >= 0:
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
    model.device = device

    H = k2.ctc_topo(
        max_token=max_token_id,
        modified=True,
        device=device,
    )
    H = k2.Fsa.from_fsas([H])

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()
    test_sets = {
        "test-clean": test_clean_cuts,
        "test-other": test_other_cuts,
    }

    for test_set, test_cut in test_sets.items():
        results_dict = decode_dataset(
            cuts=test_cut,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=H,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
