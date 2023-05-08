#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
import logging
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from model import Tdnn
from torch.nn.utils.rnn import pad_sequence

from icefall.decode import get_lattice, one_best_decoding
from icefall.utils import AttributeDict, get_texts


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
        "--words-file",
        type=str,
        required=True,
        help="Path to words.txt",
    )

    parser.add_argument("--HLG", type=str, required=True, help="Path to HLG.pt.")

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 23,
            "num_classes": 4,  # [<blk>, N, SIL, Y]
            "sample_rate": 8000,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


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


def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))
    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")

    model = Tdnn(
        num_features=params.feature_dim,
        num_classes=params.num_classes,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    logging.info(f"Loading HLG from {params.HLG}")
    HLG = k2.Fsa.from_dict(torch.load(params.HLG, map_location="cpu"))
    HLG = HLG.to(device)

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    # Note: We don't use key padding mask for attention during decoding
    with torch.no_grad():
        nnet_output = model(features)

    batch_size = nnet_output.shape[0]
    supervision_segments = torch.tensor(
        [[i, 0, nnet_output.shape[1]] for i in range(batch_size)],
        dtype=torch.int32,
    )

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
    )

    best_path = one_best_decoding(
        lattice=lattice, use_double_scores=params.use_double_scores
    )

    hyps = get_texts(best_path)
    word_sym_table = k2.SymbolTable.from_file(params.words_file)
    hyps = [[word_sym_table[i] for i in ids] for ids in hyps]

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
