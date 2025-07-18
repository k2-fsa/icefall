#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# 		         2022  Xiaomi Crop.        (authors: Mingshuang Luo)
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
Offline Usage:
(1) greedy search
./pruned_transducer_stateless5/pretrained.py \
        --checkpoint ./pruned_transducer_stateless5/exp_L_offline/pretrained.pt \
        --lang-dir ./data/lang_char \
        --method greedy_search \
        --max-sym-per-frame 1 \
        /path/to/foo.wav \
        /path/to/bar.wav
(2) modified beam search
./pruned_transducer_stateless5/pretrained.py \
        --checkpoint ./pruned_transducer_stateless5/exp_L_offline/pretrained.pt \
        --lang-dir ./data/lang_char \
        --method modified_beam_search \
        --beam-size 4 \
        /path/to/foo.wav \
        /path/to/bar.wav
(3) fast beam search
./pruned_transducer_stateless5/pretrained.py \
        --checkpoint ./pruned_transducer_stateless/exp_L_offline/pretrained.pt \
        --lang-dir ./data/lang_char \
        --method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8 \
        /path/to/foo.wav \
        /path/to/bar.wav
You can also use `./pruned_transducer_stateless5/exp_L_offline/epoch-xx.pt`.
Note: ./pruned_transducer_stateless5/exp_L_offline/pretrained.pt is generated by
./pruned_transducer_stateless5/export.py
"""


import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_params, get_transducer_model

from icefall.lexicon import Lexicon


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--lang-dir",
        type=str,
        help="""Path to lang.
        """,
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
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Used only when --method is beam_search and modified_beam_search ",
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
        default=8,
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
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame. Used only when
        --method is greedy_search.
        """,
    )
    add_model_arguments(parser)

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))

    lexicon = Lexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_transducer_model(params)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    model.device = device

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    feature_lengths = torch.tensor(feature_lengths, device=device)

    with torch.no_grad():
        encoder_out, encoder_out_lens = model.encoder(
            x=features, x_lens=feature_lengths
        )

    hyps = []
    msg = f"Using {params.decoding_method}"
    logging.info(msg)

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append([lexicon.token_table[idx] for idx in hyp])

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
