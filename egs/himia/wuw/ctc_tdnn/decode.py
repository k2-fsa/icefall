#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (Author: Weiji Zhuang,
#                                                 Liyong Guo)
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
import copy
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
from lhotse.features.io import NumpyHdf5Reader
from tqdm import tqdm

from icefall.utils import AttributeDict

from train import get_params
from graph import ctc_trivial_decoding_graph


class Arc:
    def __init__(
        self, src_state: int, dst_state: int, ilabel: int, olabel: int
    ) -> None:
        self.src_state = int(src_state)
        self.dst_state = int(dst_state)
        self.ilabel = int(ilabel)
        self.olabel = int(olabel)

    def next_state(self) -> None:
        return self.dst_state


class State:
    def __init__(self) -> None:
        self.arc_list = list()

    def add_arc(self, arc: Arc) -> None:
        self.arc_list.append(arc)


class FiniteStateTransducer:
    """Represents a decoding graph for wake word detection."""

    def __init__(self, graph: str) -> None:
        self.state_list = list()
        for arc_str in graph.split("\n"):
            arc = arc_str.strip().split()
            if len(arc) == 0:
                continue
            # 1 and 2 for final state
            # 4 for non-final state
            assert len(arc) in [1, 2, 4], f"{len(arc)} {arc_str}"
            if len(arc) == 4:  # Non-final state
                # FST must be sorted
                if len(self.state_list) <= int(arc[0]):
                    new_state = State()
                    self.state_list.append(new_state)
                self.state_list[int(arc[0])].add_arc(
                    Arc(arc[0], arc[1], arc[2], arc[3])
                )
            else:
                self.final_state_id = int(arc[0])

    def to_str(self) -> None:
        fst_str = ""
        for state_idx in range(len(self.state_list)):
            cur_state = self.state_list[state_idx]
            for arc_idx in range(len(cur_state.arc_list)):
                cur_arc = cur_state.arc_list[arc_idx]
                ilabel = cur_arc.ilabel
                olabel = cur_arc.olabel
                src_state = cur_arc.src_state
                dst_state = cur_arc.dst_state
                fst_str += f"{src_state} {dst_state} {ilabel} {olabel}\n"
        fst_str += f"{dst_state}\n"
        return fst_str


class Token:
    def __init__(self) -> None:
        self.is_active = False
        self.total_score = -float("inf")
        self.keyword_frames = 0
        self.average_keyword_score = -float("inf")
        self.average_max_keyword_score = 0.0

    def set_token(
        self,
        src_token,
        is_keyword_ilabel: bool,
        acoustic_score: float,
    ) -> None:
        """
        A dynamic programming process computing the highest score for a token
        from all possible paths which could reach this token.

        Args:
          src_token: The source token connected to current token with an arc.
          is_keyword_ilabel: If true, the arc consumes an input label which is
            a part of wake word. Otherwhise, the input label is
            blank or unknown, i.e. current token is still not part of wake word.
          acoustic_score: acoustic score of this arc.
        """

        if (
            not self.is_active
            or self.total_score < src_token.total_score + acoustic_score
        ):
            self.is_active = True
            self.total_score = src_token.total_score + acoustic_score

            if is_keyword_ilabel:
                self.average_keyword_score = (
                    acoustic_score
                    + src_token.average_keyword_score * src_token.keyword_frames
                ) / (src_token.keyword_frames + 1)

                self.keyword_frames = src_token.keyword_frames + 1
            else:
                self.average_keyword_score = 0.0


class SingleDecodable:
    def __init__(
        self,
        model_output,
        keyword_ilabel_start,
        graph,
    ):
        """
        Args:
          model_output: log_softmax(logit) with shape [T, C]
          keyword_ilabel_start: index of the first token of the wake word.
            In this recipe, tokens not for wake word has smaller token index,
            i.e. blank 0; unk 1.
          graph: decoding graph of the wake word.

        """
        self.init_token_list = [Token() for i in range(len(graph.state_list))]
        self.reset_token_list()
        self.model_output = model_output
        self.T = model_output.shape[0]
        self.utt_score = 0.0
        self.current_frame_index = 0
        self.keyword_ilabel_start = keyword_ilabel_start
        self.graph = graph
        self.number_tokens = len(self.cur_token_list)

    def reset_token_list(self) -> None:
        """
        Reset all tokens to a condition without consuming any acoustic frames.
        """
        self.cur_token_list = copy.deepcopy(self.init_token_list)
        self.expand_token_list = copy.deepcopy(self.init_token_list)
        self.cur_token_list[0].is_active = True
        self.cur_token_list[0].total_score = 0
        self.cur_token_list[0].average_keyword_score = 0

    def process_oneframe(self) -> None:
        """
        Decode a frame and update all tokens.
        """
        for state_id, cur_token in enumerate(self.cur_token_list):
            if cur_token.is_active:
                for arc_id in self.graph.state_list[state_id].arc_list:
                    acoustic_score = self.model_output[self.current_frame_index][
                        arc_id.ilabel
                    ]
                    is_keyword_ilabel = arc_id.ilabel >= self.keyword_ilabel_start
                    self.expand_token_list[arc_id.next_state()].set_token(
                        cur_token,
                        is_keyword_ilabel,
                        acoustic_score,
                    )
        # use best_score to keep total_score in a good range
        self.best_state_id = 0
        best_score = self.expand_token_list[0].total_score
        for state_id in range(self.number_tokens):
            if self.expand_token_list[state_id].is_active:
                if best_score < self.expand_token_list[state_id].total_score:
                    best_score = self.expand_token_list[state_id].total_score
                    self.best_state_id = state_id

        self.cur_token_list = self.expand_token_list
        for state_id in range(self.number_tokens):
            self.cur_token_list[state_id].total_score -= best_score
        self.expand_token_list = copy.deepcopy(self.init_token_list)
        potential_score = np.exp(
            self.cur_token_list[self.graph.final_state_id].average_keyword_score
        )
        if potential_score > self.utt_score:
            self.utt_score = potential_score
        self.current_frame_index += 1


def decode_utt(
    params: AttributeDict, utt_id: str, post_file, graph: FiniteStateTransducer
) -> Tuple[str, float]:
    """
    Decode a single utterance.

    Args:
      params:
        The return value of :func:`get_params`.
      utt_id: utt_id to be decoded, used to fetch posterior matrix from post_file.
      post_file: file to save posterior for all test set.
      graph: decoding graph.

    Returns:
      utt_id and its corresponding probability to be a wake word.
    """
    reader = NumpyHdf5Reader(post_file)
    model_output = reader.read(utt_id)
    keyword_ilabel_start = params.wakeup_word_tokens[0]
    decodable = SingleDecodable(
        model_output=model_output,
        keyword_ilabel_start=keyword_ilabel_start,
        graph=graph,
    )
    for t in range(decodable.T):
        decodable.process_oneframe()
    return utt_id, decodable.utt_score


def get_parser():
    parser = argparse.ArgumentParser(
        description="A simple FST decoder for the wake word detection\n"
    )
    parser.add_argument(
        "--decoding-graph", help="decoding graph", default="himia_ctc_graph.txt"
    )
    parser.add_argument("--post-h5", help="model output in h5 format")
    parser.add_argument("--score-file", help="file to save scores of each utterance")
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))

    keys = NumpyHdf5Reader(params.post_h5).hdf.keys()
    graph = FiniteStateTransducer(ctc_trivial_decoding_graph(params.wakeup_word_tokens))
    logging.info(f"Graph used:\n{graph.to_str()}")
    logging.info("About to load data to decoder.")
    with ProcessPoolExecutor() as executor, open(
        params.score_file, "w", encoding="utf8"
    ) as fout:
        futures = [
            executor.submit(decode_utt, params, key, params.post_h5, graph)
            for key in tqdm(keys)
        ]
        logging.info("Decoding.")
        for future in tqdm(futures):
            k, v = future.result()
            fout.write(str(k) + " " + str(v) + "\n")


if __name__ == "__main__":
    main()
