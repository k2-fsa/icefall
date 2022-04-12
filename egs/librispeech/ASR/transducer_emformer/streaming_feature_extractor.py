# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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

from typing import List, Optional

import torch
from beam_search import Hypothesis
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature


def _create_streaming_feature_extractor() -> OnlineFeature:
    """Create a CPU streaming feature extractor.

    At present, we assume it returns a fbank feature extractor with
    fixed options. In the future, we will support passing in the options
    from outside.

    Returns:
      Return a CPU streaming feature extractor.
    """
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    return OnlineFbank(opts)


class FeatureExtractionStream(object):
    def __init__(self, context_size: int, blank_id: int = 0) -> None:
        """Context size of the RNN-T decoder model."""
        self.feature_extractor = _create_streaming_feature_extractor()
        self.hyp = Hypothesis(
            ys=([blank_id] * context_size),
            log_prob=torch.tensor([0.0]),
        )  # for greedy search, will extend it to beam search

        # It contains a list of 1-D tensors representing the feature frames.
        self.feature_frames: List[torch.Tensor] = []

        self.num_fetched_frames = 0

        # For the emformer model, it contains the states of each
        # encoder layer.
        self.states: Optional[List[List[torch.Tensor]]] = None

        # For the RNN-T decoder, it contains the decoder output
        # corresponding to the decoder input self.hyp.ys[-context_size:]
        # Its shape is (decoder_out_dim,)
        self.decoder_out: Optional[torch.Tensor] = None

        # After calling `self.input_finished()`, we set this flag to True
        self._done = False

    def accept_waveform(
        self,
        sampling_rate: float,
        waveform: torch.Tensor,
    ) -> None:
        """Feed audio samples to the feature extractor and compute features
        if there are enough samples available.

        Caution:
          The range of the audio samples should match the one used in the
          training. That is, if you use the range [-1, 1] in the training, then
          the input audio samples should also be normalized to [-1, 1].

        Args
          sampling_rate:
            The sampling rate of the input audio samples. It is used for sanity
            check to ensure that the input sampling rate equals to the one
            used in the extractor. If they are not equal, then no resampling
            will be performed; instead an error will be thrown.
        waveform:
          A 1-D torch tensor of dtype torch.float32 containing audio samples.
          It should be on CPU.
        """
        self.feature_extractor.accept_waveform(
            sampling_rate=sampling_rate,
            waveform=waveform,
        )
        self._fetch_frames()

    def input_finished(self) -> None:
        """Signal that no more audio samples available and the feature
        extractor should flush the buffered samples to compute frames.
        """
        self.feature_extractor.input_finished()
        self._fetch_frames()
        self._done = True

    @property
    def done(self) -> bool:
        """Return True if `self.input_finished()` has been invoked"""
        return self._done

    def _fetch_frames(self) -> None:
        """Fetch frames from the feature extractor"""
        while self.num_fetched_frames < self.feature_extractor.num_frames_ready:
            frame = self.feature_extractor.get_frame(self.num_fetched_frames)
            self.feature_frames.append(frame)
            self.num_fetched_frames += 1
