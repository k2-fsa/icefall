# Copyright    2022  Xiaomi Corp.        (authors: Wei Kang)
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

from typing import List, Optional, Tuple

import k2
import torch

from icefall.utils import AttributeDict


class DecodeStream(object):
    def __init__(
        self,
        params: AttributeDict,
        decoding_graph: Optional[k2.Fsa] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
          decoding_graph:
            Decoding graph used for decoding, may be a TrivialGraph or a HLG.
          device:
            The device to run this stream.
        """
        if decoding_graph is not None:
            assert device == decoding_graph.device

        self.params = params

        # It contains a 2-D tensors representing the feature frames.
        self.features: torch.Tensor = None
        # how many frames are processed. (before subsampling).
        self.num_processed_frames: int = 0
        self._done: bool = False
        # The transcript of current utterance.
        self.ground_truth: str = ""
        # The decoding result (partial or final) of current utterance.
        self.hyp: List = []

        self.feature_len: int = 0

        if params.decoding_method == "greedy_search":
            self.hyp = [params.blank_id] * params.context_size
        elif params.decoding_method == "fast_beam_search":
            # feature_len is needed to get partial results.
            # The rnnt_decoding_stream for fast_beam_search.
            self.rnnt_decoding_stream: k2.RnntDecodingStream = (
                k2.RnntDecodingStream(decoding_graph)
            )
        else:
            assert (
                False
            ), f"Decoding method :{params.decoding_method} do not support"

        # The caches for streaming conformer
        # It is a List containing two tensors, the first one is the cache for
        # attention which has a shape of
        # (num_encoder_layers, left_context, encoder_dim),
        # the second one is the cache of conv_module which has a shape of
        # (num_encoder_layers, cnn_module_kernel - 1, encoder_dim).
        self.states: List[torch.Tensor] = [
            torch.zeros(
                (
                    params.num_encoder_layers,
                    params.left_context,
                    params.encoder_dim,
                ),
                device=device,
            ),
            torch.zeros(
                (
                    params.num_encoder_layers,
                    params.cnn_module_kernel - 1,
                    params.encoder_dim,
                ),
                device=device,
            ),
        ]

    @property
    def done(self) -> bool:
        """Return True if all the features are processed."""
        return self._done

    def set_features(
        self,
        features: torch.Tensor,
    ) -> None:
        """Set features tensor of current utterance."""
        self.features = features

    def get_feature_frames(self, chunk_size: int) -> Tuple[torch.Tensor, int]:
        """Consume chunk_size frames of features"""
        ret_chunk_size = min(
            self.features.size(0) - self.num_processed_frames, chunk_size + 3
        )
        ret_features = self.features[
            self.num_processed_frames : self.num_processed_frames  # noqa
            + ret_chunk_size,
            :,
        ]
        self.num_processed_frames += (
            chunk_size
            - self.params.right_context * self.params.subsampling_factor
        )

        if self.num_processed_frames >= self.features.size(0):
            self._done = True

        return ret_features, ret_chunk_size
