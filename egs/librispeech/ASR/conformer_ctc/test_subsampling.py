#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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


import torch
from subsampling import Conv2dSubsampling, VggSubsampling


def test_conv2d_subsampling():
    N = 3
    odim = 2

    for T in range(7, 19):
        for idim in range(7, 20):
            model = Conv2dSubsampling(idim=idim, odim=odim)
            x = torch.empty(N, T, idim)
            y = model(x)
            assert y.shape[0] == N
            assert y.shape[1] == ((T - 1) // 2 - 1) // 2
            assert y.shape[2] == odim


def test_vgg_subsampling():
    N = 3
    odim = 2

    for T in range(7, 19):
        for idim in range(7, 20):
            model = VggSubsampling(idim=idim, odim=odim)
            x = torch.empty(N, T, idim)
            y = model(x)
            assert y.shape[0] == N
            assert y.shape[1] == ((T - 1) // 2 - 1) // 2
            assert y.shape[2] == odim
