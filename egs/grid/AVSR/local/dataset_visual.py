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
This script is to load the visual data in GRID.
The class dataset_visual makes each visual batch data have the same shape.
"""
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .cvtransforms import HorizontalFlip, ColorNormalize


class dataset_visual(Dataset):
    def __init__(
        self,
        video_path: str,
        anno_path: str,
        file_list: str,
        vid_padding: int,
        phase: str,
    ):
        """
        Args:
          video_path:
            The dir path of the visual data.
          anno_path:
            The dir path of the texts data.
          file_list:
            The file which listing all samples for training or testing.
          vid_padding:
            The padding for each visual sample.
          phase:
            "train" or "test".
        """
        self.anno_path = anno_path
        self.vid_padding = vid_padding
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
        anno = self._load_anno(
            os.path.join(self.anno_path, spk, "align", name + ".align")
        )

        if self.phase == "train":
            vid = HorizontalFlip(vid)
        vid = ColorNormalize(vid)

        vid = self._padding(vid, self.vid_padding)

        return {
            "vid": torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
            "txt": anno.upper(),
        }

    def __len__(self):
        return len(self.data)

    def _load_vid(self, p):
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

    def _load_anno(self, name):
        with open(name, "r") as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ["SIL", "SP"], txt))
            txt = " ".join(txt)
        return txt

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
