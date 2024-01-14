#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file can be used to check if any split is corrupted.
"""

import glob
import re

import lhotse


def main():
    d = "data/fbank/XL_split_2000"
    filenames = list(glob.glob(f"{d}/cuts_XL.*.jsonl.gz"))

    pattern = re.compile(r"cuts_XL.([0-9]+).jsonl.gz")

    idx_filenames = [(int(pattern.search(c).group(1)), c) for c in filenames]

    idx_filenames = sorted(idx_filenames, key=lambda x: x[0])

    print(f"Loading {len(idx_filenames)} splits")

    s = 0
    for i, f in idx_filenames:
        cuts = lhotse.load_manifest_lazy(f)
        print(i, "filename", f)
        for i, c in enumerate(cuts):
            s += c.features.load().shape[0]
            if i > 5:
                break


if __name__ == "__main__":
    main()
