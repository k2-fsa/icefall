# Copyright      2023  Xiaomi Corp.        (authors: Yifan Yang)
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


import glob
import logging
import re
from pathlib import Path

import lhotse
from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:

            - multidataset_split_1998/multidataset/multidataset_cuts_train.*.jsonl.gz 
        """
        self.manifest_dir = Path(manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        filenames = glob.glob(
            f"{self.manifest_dir}/multidataset_split_1998/multidataset/multidataset_cuts_train.*.jsonl.gz"
        )

        pattern = re.compile(r"multidataset_cuts_train.([0-9]+).jsonl.gz")
        idx_filenames = ((int(pattern.search(f).group(1)), f) for f in filenames)
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])

        sorted_filenames = (f[1] for f in idx_filenames)

        logging.info(f"Loading {len(sorted_filenames)} splits")

        return lhotse.combine(lhotse.load_manifest_lazy(p) for p in sorted_filenames)
