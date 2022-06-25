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
./pruned_transducer_stateless2/streaming_decode.py \
        --epoch 28 \
        --avg 15 \
        --left-context 32 \
        --decode-chunk-size 8 \
        --right-context 2 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --decoding_method greedy_search \
        --num-decode-streams 1000
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
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decode_stream import DecodeStream
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import one_best_decoding
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
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
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
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
        help="""Support only greedy_search and fast_beam_search now.
        """,
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
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=16,
        help="The chunk size for decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--left-context",
        type=int,
        default=64,
        help="left context can be seen during decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--right-context",
        type=int,
        default=4,
        help="right context can be seen during decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded parallel.",
    )

    add_model_arguments(parser)

    return parser


def greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
) -> List[List[int]]:

    assert len(streams) == encoder_out.size(0)
    assert encoder_out.ndim == 3

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device
    T = encoder_out.size(1)

    decoder_input = torch.tensor(
        [stream.hyp[-context_size:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
    # decoder_out is of shape (N, decoder_out_dim)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # logging.info(f"decoder_out shape : {decoder_out.shape}")

    for t in range(T):
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        current_encoder_out = encoder_out[:, t : t + 1, :]  # noqa

        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        # logits'shape (batch_size,  vocab_size)
        logits = logits.squeeze(1).squeeze(1)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                streams[i].hyp.append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [stream.hyp[-context_size:] for stream in streams],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input,
                need_pad=False,
            )
            decoder_out = model.joiner.decoder_proj(decoder_out)

    hyp_tokens = []
    for stream in streams:
        hyp_tokens.append(stream.hyp)
    return hyp_tokens


def fast_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    processed_lens: torch.Tensor,
    decoding_streams: k2.RnntDecodingStreams,
) -> List[List[int]]:

    B, T, C = encoder_out.shape
    for t in range(T):
        # shape is a RaggedShape of shape (B, context)
        # contexts is a Tensor of shape (shape.NumElements(), context_size)
        shape, contexts = decoding_streams.get_contexts()
        # `nn.Embedding()` in torch below v1.7.1 supports only torch.int64
        contexts = contexts.to(torch.int64)
        # decoder_out is of shape (shape.NumElements(), 1, decoder_out_dim)
        decoder_out = model.decoder(contexts, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # current_encoder_out is of shape
        # (shape.NumElements(), 1, joiner_dim)
        # fmt: off
        current_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
        # fmt: on
        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        logits = logits.squeeze(1).squeeze(1)
        log_probs = logits.log_softmax(dim=-1)
        decoding_streams.advance(log_probs)

    decoding_streams.terminate_and_flush_to_streams()

    lattice = decoding_streams.format_output(processed_lens.tolist())
    best_path = one_best_decoding(lattice)
    hyp_tokens = get_texts(best_path)
    return hyp_tokens


def decode_one_chunk(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decode_streams: List[DecodeStream],
) -> List[int]:
    """Decode one chunk frames of features for each decode_streams and
    return the indexes of finished streams in a List.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decode_streams:
        A List of DecodeStream, each belonging to a utterance.
    Returns:
      Return a List containing which DecodeStreams are finished.
    """
    device = model.device

    features = []
    feature_lens = []
    states = []

    rnnt_stream_list = []
    processed_lens = []

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(
            params.decode_chunk_size * params.subsampling_factor
        )
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)
        if params.decoding_method == "fast_beam_search":
            rnnt_stream_list.append(stream.rnnt_decoding_stream)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    # if T is less than 7 there will be an error in time reduction layer,
    # because we subsample features with ((x_len - 1) // 2 - 1) // 2
    # we plus 2 here because we will cut off one frame on each size of
    # encoder_embed output as they see invalid paddings. so we need extra 2
    # frames.
    tail_length = 7 + (2 + params.right_context) * params.subsampling_factor
    if features.size(1) < tail_length:
        feature_lens += tail_length - features.size(1)
        features = torch.cat(
            [
                features,
                torch.tensor(
                    LOG_EPS, dtype=features.dtype, device=device
                ).expand(
                    features.size(0),
                    tail_length - features.size(1),
                    features.size(2),
                ),
            ],
            dim=1,
        )

    states = [
        torch.stack([x[0] for x in states], dim=2),
        torch.stack([x[1] for x in states], dim=2),
    ]
    processed_lens = torch.tensor(processed_lens, device=device)

    encoder_out, encoder_out_lens, states = model.encoder.streaming_forward(
        x=features,
        x_lens=feature_lens,
        states=states,
        left_context=params.left_context,
        right_context=params.right_context,
        processed_lens=processed_lens,
    )

    encoder_out = model.joiner.encoder_proj(encoder_out)

    if params.decoding_method == "greedy_search":
        hyp_tokens = greedy_search(model, encoder_out, decode_streams)
    elif params.decoding_method == "fast_beam_search":
        config = k2.RnntDecodingConfig(
            vocab_size=params.vocab_size,
            decoder_history_len=params.context_size,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        decoding_streams = k2.RnntDecodingStreams(rnnt_stream_list, config)
        processed_lens = processed_lens + encoder_out_lens
        hyp_tokens = fast_beam_search(
            model, encoder_out, processed_lens, decoding_streams
        )
    else:
        assert False

    states = [torch.unbind(states[0], dim=2), torch.unbind(states[1], dim=2)]

    finished_streams = []
    for i in range(len(decode_streams)):
        decode_streams[i].states = [states[0][i], states[1][i]]
        decode_streams[i].done_frames += encoder_out_lens[i]
        if params.decoding_method == "fast_beam_search":
            decode_streams[i].hyp = hyp_tokens[i]
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

    log_interval = 50

    decode_results = []
    # Contain decode streams currently running.
    decode_streams = []
    initial_states = model.encoder.get_init_state(
        params.left_context, device=device
    )
    for num, cut in enumerate(cuts):
        # each utterance has a DecodeStream.
        decode_stream = DecodeStream(
            params=params,
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
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"

        samples = torch.from_numpy(audio).squeeze(0)

        fbank = Fbank(opts)
        feature = fbank(samples.to(device))
        decode_stream.set_features(feature)
        decode_stream.ground_truth = cut.supervisions[0].text

        decode_streams.append(decode_stream)

        while len(decode_streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                params, model, sp, decode_streams
            )
            for i in sorted(finished_streams, reverse=True):
                hyp = decode_streams[i].hyp
                if params.decoding_method == "greedy_search":
                    hyp = hyp[params.context_size :]  # noqa
                decode_results.append(
                    (
                        decode_streams[i].ground_truth.split(),
                        sp.decode(hyp).split(),
                    )
                )
                del decode_streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    # decode final chunks of last sequences
    while len(decode_streams):
        finished_streams = decode_one_chunk(params, model, sp, decode_streams)
        for i in sorted(finished_streams, reverse=True):
            hyp = decode_streams[i].hyp
            if params.decoding_method == "greedy_search":
                hyp = hyp[params.context_size :]  # noqa
            decode_results.append(
                (
                    decode_streams[i].ground_truth.split(),
                    sp.decode(hyp).split(),
                )
            )
            del decode_streams[i]

    key = "greedy_search"
    if params.decoding_method == "fast_beam_search":
        key = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}"
        )
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
        # sort results so we can easily compare the difference between two
        # recognition results
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
        params.res_dir
        / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
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

    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    # for streaming
    params.suffix += f"-streaming-chunk-size-{params.decode_chunk_size}"
    params.suffix += f"-left-context-{params.left_context}"
    params.suffix += f"-right-context-{params.right_context}"

    # for fast_beam_search
    if params.decoding_method == "fast_beam_search":
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"

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
    # Decoding in streaming requires causal convolution
    params.causal_convolution = True

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

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

    model.to(device)
    model.eval()
    model.device = device

    decoding_graph = None
    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_sets = ["test-clean", "test-other"]
    test_cuts = [test_clean_cuts, test_other_cuts]

    for test_set, test_cut in zip(test_sets, test_cuts):
        results_dict = decode_dataset(
            cuts=test_cut,
            params=params,
            model=model,
            sp=sp,
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
