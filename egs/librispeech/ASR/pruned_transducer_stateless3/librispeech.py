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


class LibriSpeech:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - cuts_dev-clean.json.gz
                - cuts_dev-other.json.gz
                - cuts_test-clean.json.gz
                - cuts_test-other.json.gz
                - cuts_train-clean-100.json.gz
                - cuts_train-clean-360.json.gz
                - cuts_train-other-500.json.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_clean_100_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_train-clean-100.json.gz"
        logging.info(f"About to get train-clean-100 cuts from {f}")
        return load_manifest(f)

    def train_clean_360_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_train-clean-360.json.gz"
        logging.info(f"About to get train-clean-360 cuts from {f}")
        return load_manifest(f)

    def train_other_500_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_train-other-500.json.gz"
        logging.info(f"About to get train-other-500 cuts from {f}")
        return load_manifest(f)

    def test_clean_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_test-clean.json.gz"
        logging.info(f"About to get test-clean cuts from {f}")
        return load_manifest(f)

    def test_other_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_test-other.json.gz"
        logging.info(f"About to get test-other cuts from {f}")
        return load_manifest(f)

    def dev_clean_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_dev-clean.json.gz"
        logging.info(f"About to get dev-clean cuts from {f}")
        return load_manifest(f)

    def dev_other_cuts(self) -> CutSet:
        f = self.manifest_dir / "cuts_dev-other.json.gz"
        logging.info(f"About to get dev-other cuts from {f}")
        return load_manifest(f)
