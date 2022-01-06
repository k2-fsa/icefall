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
In this script, there are two functions:
the function HorizontalFlip is to flip the images,
the function ColorNormalize is to normalize the images.
The above two functions is to augment the images.

The input for the above functions is a sequence of images.
"""
import random


def horizontal_flip(batch_img: float, p: float):
    """
    Args:
      batch_img:
        The float array of a sequence of images, the shape of the
        arrat is (T, H, W, C).
      p:
        The probability of implementing horizontal flip, the defaults
        value is 0.5.
    Return:
      A new float array of the sequence of images after flipping.
    """
    if random.random() > p:
        batch_img = batch_img[:, :, ::-1, ...]
    return batch_img


def color_normalize(batch_img: float):
    """
    Args:
      batch_img:
        The float array of a sequence of images, the shape of the
        arrat is (T, H, W, C).
    Return:
      A new float array of the sequence of images after normalizing.
    """
    batch_img = batch_img / 255.0
    return batch_img
