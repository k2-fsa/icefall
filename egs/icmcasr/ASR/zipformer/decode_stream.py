# Copyright    2022  Xiaomi Corp.        (authors: Wei Kang,
#                                                  Zengwei Yao)
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

import math
from typing import List, Optional, Tuple

import k2
import torch
from beam_search import Hypothesis, HypothesisList

from icefall.utils import AttributeDict


class DecodeStream(object):
    def __init__(
        self,
        params: AttributeDict,
        cut_id: str,
        initial_states: List[torch.Tensor],
        decoding_graph: Optional[k2.Fsa] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
          initial_states:
            Initial decode states of the model, e.g. the return value of
            `get_init_state` in conformer.py
          decoding_graph:
            Decoding graph used for decoding, may be a TrivialGraph or a HLG.
            Used only when decoding_method is fast_beam_search.
          device:
            The device to run this stream.
        """
        if params.decoding_method == "fast_beam_search":
            assert decoding_graph is not None
            assert device == decoding_graph.device

        self.params = params
        self.cut_id = cut_id
        self.LOG_EPS = math.log(1e-10)

        self.states = initial_states

        # It contains a 2-D tensors representing the feature frames.
        self.features: torch.Tensor = None

        self.num_frames: int = 0
        # how many frames have been processed. (before subsampling).
        # we only modify this value in `func:get_feature_frames`.
        self.num_processed_frames: int = 0

        self._done: bool = False

        # The transcript of current utterance.
        self.ground_truth: str = ""

        # The decoding result (partial or final) of current utterance.
        self.hyp: List = []

        # how many frames have been processed, at encoder output
        self.done_frames: int = 0

        # The encoder_embed subsample features (T - 7) // 2
        # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
        self.pad_length = 7 + 2 * 3

        if params.decoding_method == "greedy_search":
            self.hyp = [-1] * (params.context_size - 1) + [params.blank_id]
        elif params.decoding_method == "modified_beam_search":
            self.hyps = HypothesisList()
            self.hyps.add(
                Hypothesis(
                    ys=[-1] * (params.context_size - 1) + [params.blank_id],
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                )
            )
        elif params.decoding_method == "fast_beam_search":
            # The rnnt_decoding_stream for fast_beam_search.
            self.rnnt_decoding_stream: k2.RnntDecodingStream = k2.RnntDecodingStream(
                decoding_graph
            )
        else:
            raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    @property
    def done(self) -> bool:
        """Return True if all the features are processed."""
        return self._done

    @property
    def id(self) -> str:
        return self.cut_id

    def set_features(
        self,
        features: torch.Tensor,
        tail_pad_len: int = 0,
    ) -> None:
        """Set features tensor of current utterance."""
        assert features.dim() == 2, features.dim()
        self.features = torch.nn.functional.pad(
            features,
            (0, 0, 0, self.pad_length + tail_pad_len),
            mode="constant",
            value=self.LOG_EPS,
        )
        self.num_frames = self.features.size(0)

    def get_feature_frames(self, chunk_size: int) -> Tuple[torch.Tensor, int]:
        """Consume chunk_size frames of features"""
        chunk_length = chunk_size + self.pad_length

        ret_length = min(self.num_frames - self.num_processed_frames, chunk_length)

        ret_features = self.features[
            self.num_processed_frames : self.num_processed_frames + ret_length  # noqa
        ]

        self.num_processed_frames += chunk_size
        if self.num_processed_frames >= self.num_frames:
            self._done = True

        return ret_features, ret_length

    def decoding_result(self) -> List[int]:
        """Obtain current decoding result."""
        if self.params.decoding_method == "greedy_search":
            return self.hyp[self.params.context_size :]  # noqa
        elif self.params.decoding_method == "modified_beam_search":
            best_hyp = self.hyps.get_most_probable(length_norm=True)
            return best_hyp.ys[self.params.context_size :]  # noqa
        else:
            assert self.params.decoding_method == "fast_beam_search"
            return self.hyp
