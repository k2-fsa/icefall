# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang,
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


class Stream(object):
    def __init__(
        self,
        params: AttributeDict,
        cut_id: str,
        decoding_graph: Optional[k2.Fsa] = None,
        device: torch.device = torch.device("cpu"),
        LOG_EPS: float = math.log(1e-10),
    ) -> None:
        """
        Args:
          params:
            It's the return value of :func:`get_params`.
          decoding_graph:
            The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
            only when --decoding_method is fast_beam_search.
          device:
            The device to run this stream.
        """
        self.LOG_EPS = LOG_EPS
        self.cut_id = cut_id

        # Containing attention caches and convolution caches
        self.states: Optional[
            Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
        ] = None

        # It uses different attributes for different decoding methods.
        self.context_size = params.context_size
        self.decoding_method = params.decoding_method
        if params.decoding_method == "greedy_search":
            self.hyp = [params.blank_id] * params.context_size
        elif params.decoding_method == "modified_beam_search":
            self.hyps = HypothesisList()
            self.hyps.add(
                Hypothesis(
                    ys=[params.blank_id] * params.context_size,
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                )
            )
        elif params.decoding_method == "fast_beam_search":
            # feature_len is needed to get partial results.
            # The rnnt_decoding_stream for fast_beam_search.
            self.rnnt_decoding_stream: k2.RnntDecodingStream = k2.RnntDecodingStream(
                decoding_graph
            )
            self.hyp: Optional[List[int]] = None
        else:
            raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

        self.ground_truth: str = ""

        self.feature: Optional[torch.Tensor] = None
        # Make sure all feature frames can be used.
        # Add 2 here since we will drop the first and last after subsampling.
        self.chunk_length = params.chunk_length
        self.pad_length = (
            params.right_context_length + 2 * params.subsampling_factor + 3
        )
        self.num_frames = 0
        self.num_processed_frames = 0

        # After all feature frames are processed, we set this flag to True
        self._done = False

    def set_feature(self, feature: torch.Tensor) -> None:
        assert feature.dim() == 2, feature.dim()
        self.num_frames = feature.size(0)
        # tail padding
        self.feature = torch.nn.functional.pad(
            feature,
            (0, 0, 0, self.pad_length),
            mode="constant",
            value=self.LOG_EPS,
        )

    def set_ground_truth(self, ground_truth: str) -> None:
        self.ground_truth = ground_truth

    def set_states(
        self, states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
    ) -> None:
        """Set states."""
        self.states = states

    def get_feature_chunk(self) -> torch.Tensor:
        """Get a chunk of feature frames.

        Returns:
          A tensor of shape (ret_length, feature_dim).
        """
        update_length = min(
            self.num_frames - self.num_processed_frames, self.chunk_length
        )
        ret_length = update_length + self.pad_length

        ret_feature = self.feature[
            self.num_processed_frames : self.num_processed_frames + ret_length
        ]
        # Cut off used frames.
        # self.feature = self.feature[update_length:]

        self.num_processed_frames += update_length
        if self.num_processed_frames >= self.num_frames:
            self._done = True

        return ret_feature

    @property
    def done(self) -> bool:
        """Return True if all feature frames are processed."""
        return self._done

    @property
    def id(self) -> str:
        return self.cut_id

    def decoding_result(self) -> List[int]:
        """Obtain current decoding result."""
        if self.decoding_method == "greedy_search":
            return self.hyp[self.context_size :]
        elif self.decoding_method == "modified_beam_search":
            best_hyp = self.hyps.get_most_probable(length_norm=True)
            return best_hyp.ys[self.context_size :]
        else:
            assert self.decoding_method == "fast_beam_search"
            return self.hyp
