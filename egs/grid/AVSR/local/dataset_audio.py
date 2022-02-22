#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Mingshuang Luo)
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
This script is to load the audio data in GRID.
The class dataset_audio makes each audio batch data have the same shape.
"""
import numpy as np
import os

import torch
import torchaudio
from torch.utils.data import Dataset

import kaldifeat


class AudioDataSet(Dataset):
    def __init__(
        self,
        video_path: str,
        anno_path: str,
        file_list: str,
        aud_padding: int,
        sample_rate: int,
        feature_dim: int,
        phase: str,
    ) -> None:
        """
        Args:
          video_path:
            The dir path of the visual data.
          anno_path:
            The dir path of the texts data.
          file_list:
            A txt file which listing all samples for training or testing.
          aud_padding:
            The padding for each audio sample.
          sample_rate:
            The sample rate for extracting fbank feature.
          feature_dim:
            The dim for fbank feature.
          phase:
            "train" or "test".
        """
        self.anno_path = anno_path
        self.aud_padding = aud_padding
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.phase = phase

        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = self.feature_dim
        self.fbank = kaldifeat.Fbank(opts)

        with open(file_list, "r") as f:
            self.videos = [
                os.path.join(video_path, line.strip()) for line in f.readlines()
            ]

        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)
            aud = (
                vid.replace("lip", "audio_25k").replace("/video/mpg_6000", "")
                + ".wav"
            )
            self.data.append((aud, items[-4], items[-1]))

    def __getitem__(self, idx):
        (aud, spk, name) = self.data[idx]
        aud = self._load_aud(aud)
        aud = self._padding(aud, self.aud_padding)
        anno = self._load_anno(
            os.path.join(self.anno_path, spk, "align", name + ".align")
        )

        return {
            "aud": torch.FloatTensor(aud),
            "txt": anno.upper(),
        }

    def __len__(self):
        return len(self.data)

    def _load_aud(self, filename):
        """Load the audio data.
        Args:
            filename:
                The full path of a wav file.
        Return:
            The fbank feature array.
        """
        wave, _ = torchaudio.load(filename)
        wave = wave[0]
        features = self.fbank(wave)
        return features

    def _load_anno(self, name):
        """Load the text file.
        Args:
            name:
                The file which records the text.
        Return:
            A sequence of words.
        """
        with open(name, "r") as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ["SIL", "SP"], txt))
            txt = " ".join(txt)
        return txt

    def _padding(self, array, length):
        """Pad zeros for the feature array.
        Args:
            array:
                The feature arry. (Audio or Visual feature)
            length:
                The length for padding.
        Return:
            A new feature array after padding.
        """
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
