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
This script is to load the pair of audio-visual data in GRID.
The class dataset_av makes each audio-visual batch data have the same shape.
"""
import cv2
import numpy as np
import os

import torch
import torchaudio
from torch.utils.data import Dataset

import kaldifeat
from .cvtransforms import horizontal_flip, color_normalize


class AudioVisualDataset(Dataset):
    def __init__(
        self,
        video_path,
        anno_path,
        file_list,
        feature_dim,
        vid_pading,
        aud_pading,
        sample_rate,
        phase,
    ):
        """
        Args:
          video_path:
            The dir path of the visual data.
          anno_path:
            The dir path of the texts data.
          file_list:
            The file which listing all samples for training or testing.
          feature_dim:
            The dimension for fbank feature.
          vid_padding:
            The padding for each visual sample.
          aud_padding:
            The padding for each audio sample.
          sample_rate:
            The sample rate for extracting fbank feature.
          phase:
            "train" or "test".
        """
        self.anno_path = anno_path
        self.vid_pading = vid_pading
        self.aud_pading = aud_pading
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate
        self.phase = phase
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
            self.data.append((vid, aud, items[-4], items[-1]))

    def __getitem__(self, idx):
        (vid, aud, spk, name) = self.data[idx]

        vid = self._load_vid(vid)
        aud = self._load_aud(aud)
        vid = self._padding(vid, self.vid_pading)
        aud = self._padding(aud, self.aud_pading)
        anno = self._load_anno(
            os.path.join(self.anno_path, spk, "align", name + ".align")
        )

        if self.phase == "train":
            vid = horizontal_flip(vid)
        vid = color_normalize(vid)

        vid = self._padding(vid, self.vid_pading)
        aud = self._padding(aud, self.aud_pading)

        return {
            "vid": torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
            "aud": torch.FloatTensor(aud),
            "txt": anno.upper(),
        }

    def __len__(self):
        return len(self.data)

    def _load_vid(self, p):
        """Load the visual data.
        Args:
            p:
                A directory which contains a sequence of frames
            for a visual sample.
        Return:
            The array of a visual sample.
        """
        files = os.listdir(p)
        files = list(filter(lambda file: file.find(".jpg") != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: im is not None, array))
        array = [
            cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            for im in array
        ]
        array = np.stack(array, axis=0).astype(np.float32)
        return array

    def _load_aud(self, filename):
        """Load the audio data.
        Args:
            filename:
                The full path of a wav file.
        Return:
            The fbank feature array.
        """
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = self.feature_dim
        fbank = kaldifeat.Fbank(opts)
        wave, sample_rate = torchaudio.load(filename)
        features = fbank(wave[0])

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
