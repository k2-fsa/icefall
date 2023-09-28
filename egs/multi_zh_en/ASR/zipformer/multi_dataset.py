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


import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

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

        # LibriSpeech
        train_clean_100_cuts = self.train_clean_100_cuts()
        train_clean_360_cuts = self.train_clean_360_cuts()
        train_other_500_cuts = self.train_other_500_cuts()

        return CutSet.mux(
            aishell_2_cuts,
            train_clean_100_cuts,
            train_clean_360_cuts,
            train_other_500_cuts,
            weights=[
                len(aishell_2_cuts),
                len(train_clean_100_cuts),
                len(train_clean_360_cuts),
                len(train_other_500_cuts),
            ],
        )

    def dev_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # AISHELL-2
        logging.info("Loading Aishell-2 DEV set in lazy mode")
        aishell2_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_dev.jsonl.gz"
        )

        # LibriSpeech
        dev_clean_cuts = self.dev_clean_cuts()
        dev_other_cuts = self.dev_other_cuts()

        return CutSet.mux(
            aishell2_dev_cuts,
            dev_clean_cuts,
            dev_other_cuts,
            weights=[
                len(aishell2_dev_cuts),
                len(dev_clean_cuts),
                len(dev_other_cuts),
            ],
        )

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

        # LibriSpeech
        test_clean_cuts = self.test_clean_cuts()
        test_other_cuts = self.test_other_cuts()
        
        logging.info("Loading TAL-CSASR set in lazy mode")
        tal_csasr_cuts = load_manifest_lazy(
            self.fbank_dir / "tal_csasr_cuts_test_set.jsonl.gz"
        )

        return {
            "aishell-2_test": aishell2_test_cuts,
            "aishell-2_dev": aishell2_dev_cuts,
            "librispeech_test_clean": test_clean_cuts,
            "librispeech_test_other": test_other_cuts,
            "tal_csasr_cuts_test": tal_csasr_cuts,
        }

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_test-other.jsonl.gz"
        )
