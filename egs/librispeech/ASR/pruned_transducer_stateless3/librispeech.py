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

from lhotse import CutSet, load_manifest_lazy


class LibriSpeech:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - librispeech_cuts_dev-clean.jsonl.gz
                - librispeech_cuts_dev-other.jsonl.gz
                - librispeech_cuts_test-clean.jsonl.gz
                - librispeech_cuts_test-other.jsonl.gz
                - librispeech_cuts_train-clean-100.jsonl.gz
                - librispeech_cuts_train-clean-360.jsonl.gz
                - librispeech_cuts_train-other-500.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_clean_100_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        logging.info(f"About to get train-clean-100 cuts from {f}")
        return load_manifest_lazy(f)

    def train_clean_360_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        logging.info(f"About to get train-clean-360 cuts from {f}")
        return load_manifest_lazy(f)

    def train_other_500_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        logging.info(f"About to get train-other-500 cuts from {f}")
        return load_manifest_lazy(f)

    def test_clean_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        logging.info(f"About to get test-clean cuts from {f}")
        return load_manifest_lazy(f)

    def test_other_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        logging.info(f"About to get test-other cuts from {f}")
        return load_manifest_lazy(f)

    def dev_clean_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        logging.info(f"About to get dev-clean cuts from {f}")
        return load_manifest_lazy(f)

    def dev_other_cuts(self) -> CutSet:
        f = self.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
        logging.info(f"About to get dev-other cuts from {f}")
        return load_manifest_lazy(f)

    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )
