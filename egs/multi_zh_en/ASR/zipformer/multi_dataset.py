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
    def __init__(self, fbank_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - aishell2_cuts_train.jsonl.gz
        """
        self.fbank_dir = Path(fbank_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # AISHELL-2
        logging.info("Loading Aishell-2 in lazy mode")
        aishell_2_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_train.jsonl.gz"
        )

        return CutSet.mux(
            aishell_2_cuts,
            weights=[
                len(aishell_2_cuts),
            ],
        )

    def dev_cuts(self) -> List[CutSet]:
        logging.info("About to get multidataset dev cuts")

        # AISHELL-2
        logging.info("Loading Aishell-2 DEV set in lazy mode")
        aishell2_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_dev.jsonl.gz"
        )

        return [
            aishell2_dev_cuts,
        ]

    def test_cuts(self) -> Dict[str, CutSet]:
        logging.info("About to get multidataset test cuts")

        # AISHELL-2
        logging.info("Loading Aishell-2 set in lazy mode")
        aishell2_test_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_test.jsonl.gz"
        )
        aishell2_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_dev.jsonl.gz"
        )

        return {
            "aishell-2_test": aishell2_test_cuts,
            "aishell-2_dev": aishell2_dev_cuts,
        }
