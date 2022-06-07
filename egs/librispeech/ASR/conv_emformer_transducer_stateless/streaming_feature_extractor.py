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
from beam_search import HypothesisList
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature

from icefall.utils import AttributeDict


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


class Stream(object):
    def __init__(
        self,
        params: AttributeDict,
        audio_sample: torch.Tensor,
        ground_truth: str,
        device: torch.device = torch.devive("cpu"),
    ) -> None:
        """
        Args:
          context_size:
            Context size of the RNN-T decoder model.
          decoding_method:
            Decoding method. The possible values are:
              - greedy_search
              - modified_beam_search
        """
        self.feature_extractor = _create_streaming_feature_extractor()
        # It contains a list of 1-D tensors representing the feature frames.
        self.feature_frames: List[torch.Tensor] = []
        self.num_fetched_frames = 0

        # After calling `self.input_finished()`, we set this flag to True
        self._done = False

        # Initailize zero states.
        past_len: int = 0
        attn_caches = [
            [
                torch.zeros(params.memory_size, params.d_model, device=device),
                torch.zeros(
                    params.left_context_length, params.d_model, device=device
                ),
                torch.zeros(
                    params.left_context_length, params.d_model, device=device
                ),
            ]
            for _ in range(params.num_encoder_layers)
        ]
        conv_caches = [
            torch.zeros(params.d_model, params.cnn_module_kernel, device=device)
            for _ in range(params.num_encoder_layers)
        ]
        self.states = [past_len, attn_caches, conv_caches]

        # It use different attributes for different decoding methods.
        self.context_size = params.context_size
        self.decoding_method = params.decoding_method
        if params.decoding_method == "greedy_search":
            self.hyp: Optional[List[int]] = None
            self.decoder_out: Optional[torch.Tensor] = None
        elif params.decoding_method == "modified_beam_search":
            self.hyps = HypothesisList()
        else:
            raise ValueError(
                f"Unsupported decoding method: {params.decoding_method}"
            )

        self.sample_rate = params.sample_rate
        self.audio_sample = audio_sample
        # Current index of sample
        self.cur_index = 0

        self.ground_truth = ground_truth

    def accept_waveform(
        self,
        # sampling_rate: float,
        # waveform: torch.Tensor,
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
        start = self.cur_index
        end = self.cur_index + 1024
        waveform = self.audio_sample[start:end]
        self.cur_index = end

        self.feature_extractor.accept_waveform(
            sampling_rate=self.sampling_rate,
            waveform=waveform,
        )
        self._fetch_frames()

        if waveform.numel() == 0:
            self.input_finished()

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

    def decoding_result(self) -> List[int]:
        """Obtain current decoding result."""
        if self.decoding_method == "greedy_search":
            return self.hyp[self.context_size :]
        else:
            assert self.decoding_method == "modified_beam_search"
            best_hyp = self.hyps.get_most_probable(length_norm=True)
            return best_hyp.ys[self.context_size :]
