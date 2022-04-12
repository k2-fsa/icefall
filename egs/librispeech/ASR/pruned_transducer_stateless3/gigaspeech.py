# Copyright      2021  Piotr Żelasko
#                2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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


import logging
from pathlib import Path

from lhotse import CutSet, load_manifest


class GigaSpeech:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - cuts_XL_raw.jsonl.gz
                - cuts_L_raw.jsonl.gz
                - cuts_M_raw.jsonl.gz
                - cuts_S_raw.jsonl.gz
                - cuts_XS_raw.jsonl.gz
                - cuts_DEV_raw.jsonl.gz
                - cuts_TEST_raw.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_XL_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_XL_raw.jsonl.gz"
        logging.info(f"About to get train-XL cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_L_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_L_raw.jsonl.gz"
        logging.info(f"About to get train-L cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_M_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_M_raw.jsonl.gz"
        logging.info(f"About to get train-M cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_S_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_S_raw.jsonl.gz"
        logging.info(f"About to get train-S cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_XS_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_XS_raw.jsonl.gz"
        logging.info(f"About to get train-XS cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def test_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_TEST.jsonl.gz"
        logging.info(f"About to get TEST cuts from {f}")
        return load_manifest(f)

    def dev_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_DEV.jsonl.gz"
        logging.info(f"About to get DEV cuts from {f}")
        return load_manifest(f)
