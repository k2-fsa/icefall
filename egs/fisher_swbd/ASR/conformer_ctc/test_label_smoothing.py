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

from distutils.version import LooseVersion

import torch
from label_smoothing import LabelSmoothingLoss

torch_ver = LooseVersion(torch.__version__)


def test_with_torch_label_smoothing_loss():
    if torch_ver < LooseVersion("1.10.0"):
        print(f"Current torch version: {torch_ver}")
        print("Please use torch >= 1.10 to run this test - skipping")
        return
    torch.manual_seed(20211105)
    x = torch.rand(20, 30, 5000)
    tgt = torch.randint(low=-1, high=x.size(-1), size=x.shape[:2])
    for reduction in ["none", "sum", "mean"]:
        custom_loss_func = LabelSmoothingLoss(
            ignore_index=-1, label_smoothing=0.1, reduction=reduction
        )
        custom_loss = custom_loss_func(x, tgt)

        torch_loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=-1, reduction=reduction, label_smoothing=0.1
        )
        torch_loss = torch_loss_func(x.reshape(-1, x.size(-1)), tgt.reshape(-1))
        assert torch.allclose(custom_loss, torch_loss)


def main():
    test_with_torch_label_smoothing_loss()


if __name__ == "__main__":
    main()
