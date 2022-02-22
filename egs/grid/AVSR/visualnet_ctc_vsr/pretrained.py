#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Wei Kang
#                                                    Mingshuang Luo)
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
import cv2
import logging
import numpy as np
import os

import k2
import torch
from model import VisualNet

from icefall.decode import (
    get_lattice,
    one_best_decoding,
    rescore_with_whole_lattice,
)
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

    parser.add_argument(
        "--HLG", type=str, required=True, help="Path to HLG.pt."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method.
        Possible values are:
        (1) 1best - Use the best path as decoding output. Only
            the transformer encoder output is used for decoding.
            We call it HLG decoding.
        (2) whole-lattice-rescoring - Use an LM to rescore the
            decoding lattice and then use 1best to decode the
            rescored lattice.
            We call it HLG decoding + n-gram LM rescoring.
        """,
    )

    parser.add_argument(
        "--G",
        type=str,
        help="""An LM for rescoring.
        Used only when method is
        whole-lattice-rescoring.
        It's usually a 4-gram LM.
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.1,
        help="""
        Used only when method is whole-lattice-rescoring.
        It specifies the scale for n-gram LM scores.
        (Note: You need to tune it on a dataset.)
        """,
    )

    parser.add_argument(
        "--lipframes-dirs",
        type=str,
        nargs="+",
        help="The input visual file(s) to transcribe. "
        "Supported formats are those supported by cv2.imread(). "
        "The frames sample rate is 25fps.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "num_classes": 28,
            "search_beam": 20,
            "output_beam": 5,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


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
    model = VisualNet(num_classes=params.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    logging.info(f"Loading HLG from {params.HLG}")
    HLG = k2.Fsa.from_dict(torch.load(params.HLG, map_location="cpu"))
    HLG = HLG.to(device)
    if not hasattr(HLG, "lm_scores"):
        # For whole-lattice-rescoring and attention-decoder
        HLG.lm_scores = HLG.scores.clone()

    if params.method == "whole-lattice-rescoring":
        logging.info(f"Loading G from {params.G}")
        G = k2.Fsa.from_dict(torch.load(params.G, map_location="cpu"))
        # Add epsilon self-loops to G as we will compose
        # it with the whole lattice later
        G = G.to(device)
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        G.lm_scores = G.scores.clone()

    logging.info("Loading lip roi frames")

    vid = []
    for sample_dir in params.lipframes_dirs:
        files = os.listdir(sample_dir)
        files = list(filter(lambda file: file.find(".jpg") != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(sample_dir, file)) for file in files]
        array = list(filter(lambda im: im is not None, array))
        array = [
            cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            for im in array
        ]
        array = np.stack(array, axis=0).astype(np.float32)
        vid.append(array)

    _, H, W, C = vid[0].shape
    features = torch.zeros(len(vid), 75, H, W, C).to(device)
    for i in range(len(vid)):
        length = vid[i].shape[0]
        features[i][:length] = torch.FloatTensor(vid[i]).to(device)

    logging.info("Decoding started")
    features = features / 255.0
    with torch.no_grad():
        nnet_output = model(features.permute(0, 4, 1, 2, 3))
        # nnet_output is (N, T, C)

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

    if params.method == "1best":
        logging.info("Use HLG decoding")
        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
    elif params.method == "whole-lattice-rescoring":
        logging.info("Use HLG decoding + LM rescoring")
        best_path_dict = rescore_with_whole_lattice(
            lattice=lattice,
            G_with_epsilon_loops=G,
            lm_scale_list=[params.ngram_lm_scale],
        )
        best_path = next(iter(best_path_dict.values()))

    hyps = get_texts(best_path)
    word_sym_table = k2.SymbolTable.from_file(params.words_file)
    hyps = [[word_sym_table[i] for i in ids] for ids in hyps]

    s = "\n"
    for filename, hyp in zip(params.lipframes_dirs, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
