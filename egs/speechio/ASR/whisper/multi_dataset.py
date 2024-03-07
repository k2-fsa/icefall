# Copyright      2023  Xiaomi Corp.        (authors: Zengrui Jin)
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
from typing import Dict, List

import lhotse
from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, fbank_dir: str, start_index: int = 0, end_index: int = 26):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - speechio_cuts_SPEECHIO_ASR_ZH00000.jsonl.gz
            ...
            - speechio_cuts_SPEECHIO_ASR_ZH00026.jsonl.gz
        """
        self.fbank_dir = Path(fbank_dir)
        self.start_index = start_index
        self.end_index = end_index

    def test_cuts(self) -> Dict[str, CutSet]:
        logging.info("About to get multidataset test cuts")

        dataset_parts = []
        for i in range(self.start_index, self.end_index + 1):
            idx = f"{i}".zfill(2)
            dataset_parts.append(f"SPEECHIO_ASR_ZH000{idx}")

        prefix = "speechio"
        suffix = "jsonl.gz"

        results_dict = {}
        for partition in dataset_parts:
            path = f"{prefix}_cuts_{partition}.{suffix}"

            logging.info(f"Loading {path} set in lazy mode")
            test_cuts = load_manifest_lazy(self.fbank_dir / path)
            results_dict[partition] = test_cuts

        return results_dict
