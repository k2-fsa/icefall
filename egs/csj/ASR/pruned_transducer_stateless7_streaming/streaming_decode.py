#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Authors: Wei Kang, Fangjun Kuang)
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
./pruned_transducer_stateless7_streaming/streaming_decode.py \
  --epoch 28 \
  --avg 15 \
  --decode-chunk-len 32 \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --decoding_method greedy_search \
  --lang data/lang_char \
  --num-decode-streams 2000
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import torch
import torch.nn as nn
from asr_datamodule import CSJAsrDataModule
from decode import save_results
from decode_stream import DecodeStream
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet
from streaming_beam_search import (
    fast_beam_search_one_best,
    greedy_search,
    modified_beam_search,
)
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_params, get_transducer_model
from zipformer import stack_states, unstack_states

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, setup_logger, str2bool

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 0.
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
        "--gpu",
        type=int,
        default=0,
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
        default="pruned_transducer_stateless2/exp",
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
        "--decoding-graph",
        type=str,
        default="",
        help="""Used only when --decoding-method is
        fast_beam_search""",
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
        default=4.0,
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

    parser.add_argument(
        "--res-dir",
        type=Path,
        default=None,
        help="The path to save results.",
    )

    add_model_arguments(parser)

    return parser


def decode_one_chunk(
    params: AttributeDict,
    model: nn.Module,
    decode_streams: List[DecodeStream],
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

    features = []
    feature_lens = []
    states = []
    processed_lens = []

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(params.decode_chunk_len)
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    # We subsample features with ((x_len - 7) // 2 + 1) // 2 and the max downsampling
    # factor in encoders is 8.
    # After feature embedding (x_len - 7) // 2, we have (23 - 7) // 2 = 8.
    tail_length = 23
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
    processed_lens = torch.tensor(processed_lens, device=device)

    encoder_out, encoder_out_lens, new_states = model.encoder.streaming_forward(
        x=features,
        x_lens=feature_lens,
        states=states,
    )

    encoder_out = model.joiner.encoder_proj(encoder_out)

    if params.decoding_method == "greedy_search":
        greedy_search(model=model, encoder_out=encoder_out, streams=decode_streams)
    elif params.decoding_method == "fast_beam_search":
        processed_lens = processed_lens + encoder_out_lens
        fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            streams=decode_streams,
            beam=params.beam,
            max_states=params.max_states,
            max_contexts=params.max_contexts,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            streams=decode_streams,
            encoder_out=encoder_out,
            num_active_paths=params.num_active_paths,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    states = unstack_states(new_states)

    finished_streams = []
    for i in range(len(decode_streams)):
        decode_streams[i].states = states[i]
        decode_streams[i].done_frames += encoder_out_lens[i]
        if decode_streams[i].done:
            finished_streams.append(i)

    return finished_streams


def decode_dataset(
    cuts: CutSet,
    params: AttributeDict,
    model: nn.Module,
    sp: Tokenizer,
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
    opts.mel_opts.high_freq = -400

    log_interval = 50

    decode_results = []
    # Contain decode streams currently running.
    decode_streams = []
    for num, cut in enumerate(cuts):
        # each utterance has a DecodeStream.
        initial_states = model.encoder.get_init_state(device=device)
        decode_stream = DecodeStream(
            params=params,
            cut_id=cut.id,
            initial_states=initial_states,
            decoding_graph=decoding_graph,
            device=device,
        )

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        # - this is to avoid sending [-32k,+32k] signal in...
        # - some lhotse AudioTransform classes can make the signal
        #   be out of range [-1, 1], hence the tolerance 10
        assert (
            np.abs(audio).max() <= 10
        ), "Should be normalized to [-1, 1], 10 for tolerance..."

        samples = torch.from_numpy(audio).squeeze(0)

        fbank = Fbank(opts)
        feature = fbank(samples.to(device))
        decode_stream.set_features(feature, tail_pad_len=params.decode_chunk_len)
        decode_stream.ground_truth = cut.supervisions[0].custom[params.transcript_mode]

        decode_streams.append(decode_stream)

        while len(decode_streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                params=params, model=model, decode_streams=decode_streams
            )
            for i in sorted(finished_streams, reverse=True):
                decode_results.append(
                    (
                        decode_streams[i].id,
                        sp.text2word(decode_streams[i].ground_truth),
                        sp.text2word(sp.decode(decode_streams[i].decoding_result())),
                    )
                )
                del decode_streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    # decode final chunks of last sequences
    while len(decode_streams):
        finished_streams = decode_one_chunk(
            params=params, model=model, decode_streams=decode_streams
        )
        for i in sorted(finished_streams, reverse=True):
            decode_results.append(
                (
                    decode_streams[i].id,
                    sp.text2word(decode_streams[i].ground_truth),
                    sp.text2word(sp.decode(decode_streams[i].decoding_result())),
                )
            )
            del decode_streams[i]

    if params.decoding_method == "greedy_search":
        key = "greedy_search"
    elif params.decoding_method == "fast_beam_search":
        key = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}"
        )
    elif params.decoding_method == "modified_beam_search":
        key = f"num_active_paths_{params.num_active_paths}"
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")
    return {key: decode_results}


@torch.no_grad()
def main():
    parser = get_parser()
    CSJAsrDataModule.add_arguments(parser)
    Tokenizer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    if not params.res_dir:
        params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    # for streaming
    params.suffix += f"-streaming-chunk-size-{params.decode_chunk_len}"

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
        device = torch.device("cuda", params.gpu)

    logging.info(f"Device: {device}")

    sp = Tokenizer.load(params.lang, params.lang_type)

    # <blk> and <unk> is defined in local/prepare_lang_char.py
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

    decoding_graph = None
    if params.decoding_graph:
        decoding_graph = k2.Fsa.from_dict(
            torch.load(params.decoding_graph, map_location=device)
        )
    elif params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    csj_corpus = CSJAsrDataModule(args)

    for subdir in ["eval1", "eval2", "eval3", "excluded", "valid"]:
        results_dict = decode_dataset(
            cuts=getattr(csj_corpus, f"{subdir}_cuts")(),
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
        )
        tot_err = save_results(
            params=params, test_set_name=subdir, results_dict=results_dict
        )

        with (
            params.res_dir
            / (
                f"{subdir}-{params.decode_chunk_len}"
                f"_{params.avg}_{params.epoch}.cer"
            )
        ).open("w") as fout:
            if len(tot_err) == 1:
                fout.write(f"{tot_err[0][1]}")
            else:
                fout.write("\n".join(f"{k}\t{v}") for k, v in tot_err)

    logging.info("Done!")


if __name__ == "__main__":
    main()
