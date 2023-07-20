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

import lhotse
from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, fbank_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - aidatatang_cuts_train.jsonl.gz
            - aishell_cuts_train.jsonl.gz
            - aishell2_cuts_train.jsonl.gz
            - aishell4_cuts_train_L.jsonl.gz
            - aishell4_cuts_train_M.jsonl.gz
            - aishell4_cuts_train_S.jsonl.gz
            - alimeeting-far_cuts_train.jsonl.gz
            - cuts_L.jsonl.gz
            - cuts_M.jsonl.gz
            - cuts_S.jsonl.gz
            - primewords_cuts_train.jsonl.gz
            - stcmds_cuts_train.jsonl.gz
            - thchs_30_cuts_train.jsonl.gz
        """
        self.fbank_dir = Path(fbank_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # LibriSpeech
        logging.info("Loading LibriSpeech in lazy mode")
        librispeech_cuts = load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

        # GigaSpeech
        filenames = glob.glob(f"{self.fbank_dir}/XL_split/cuts_XL.*.jsonl.gz")

        pattern = re.compile(r"cuts_XL.([0-9]+).jsonl.gz")
        idx_filenames = ((int(pattern.search(f).group(1)), f) for f in filenames)
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])

        sorted_filenames = [f[1] for f in idx_filenames]

        logging.info(f"Loading GigaSpeech {len(sorted_filenames)} splits in lazy mode")

        gigaspeech_cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in sorted_filenames
        )

        # CommonVoice
        logging.info("Loading CommonVoice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(
            self.fbank_dir / f"cv-en_cuts_train.jsonl.gz"
        )

        # LibriHeavy
        logging.info("Loading LibriHeavy in lazy mode")
        libriheavy_small_cuts = load_manifest_lazy(
            self.fbank_dir / "libriheavy_cuts_train_small.jsonl.gz"
        )
        libriheavy_medium_cuts = load_manifest_lazy(
            self.fbank_dir / "libriheavy_cuts_train_medium.jsonl.gz"
        )
        libriheavy_cuts = lhotse.combine(libriheavy_small_cuts, libriheavy_medium_cuts)

        return CutSet.mux(
            librispeech_cuts,
            gigaspeech_cuts,
            commonvoice_cuts,
            libriheavy_cuts,
            weights=[
                len(librispeech_cuts),
                len(gigaspeech_cuts),
                len(commonvoice_cuts),
                len(libriheavy_cuts),
            ],
        )

    def test_cuts(self) -> CutSet:
        logging.info("About to get multidataset test cuts")

        # GigaSpeech
        logging.info("Loading GigaSpeech DEV in lazy mode")
        gigaspeech_dev_cuts = load_manifest_lazy(self.fbank_dir / "cuts_DEV.jsonl.gz")

        logging.info("Loading GigaSpeech TEST in lazy mode")
        gigaspeech_test_cuts = load_manifest_lazy(self.fbank_dir / "cuts_TEST.jsonl.gz")

        # CommonVoice
        logging.info("Loading CommonVoice DEV in lazy mode")
        commonvoice_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "cv-en_cuts_dev.jsonl.gz"
        )

        logging.info("Loading CommonVoice TEST in lazy mode")
        commonvoice_test_cuts = load_manifest_lazy(
            self.fbank_dir / "cv-en_cuts_test.jsonl.gz"
        )

        return [
            gigaspeech_dev_cuts,
            gigaspeech_test_cuts,
            commonvoice_dev_cuts,
            commonvoice_test_cuts,
        ]