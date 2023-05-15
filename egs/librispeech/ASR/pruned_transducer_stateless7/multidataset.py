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
    def __init__(self, manifest_dir: str, cv_manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:

            - librispeech_cuts_train-all-shuf.jsonl.gz
            - gigaspeech_XL_split_2000/gigaspeech_cuts_XL.*.jsonl.gz

          cv_manifest_dir:
            It is expected to contain the following files:

            - cv-en_cuts_train.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)
        self.cv_manifest_dir = Path(cv_manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # LibriSpeech
        logging.info(f"Loading LibriSpeech in lazy mode")
        librispeech_cuts = load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

        # GigaSpeech
        filenames = glob.glob(
            f"{self.manifest_dir}/gigaspeech_XL_split_2000/gigaspeech_cuts_XL.*.jsonl.gz"
        )

        pattern = re.compile(r"gigaspeech_cuts_XL.([0-9]+).jsonl.gz")
        idx_filenames = ((int(pattern.search(f).group(1)), f) for f in filenames)
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])

        sorted_filenames = [f[1] for f in idx_filenames]

        logging.info(f"Loading GigaSpeech {len(sorted_filenames)} splits in lazy mode")

        gigaspeech_cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in sorted_filenames
        )

        # CommonVoice
        logging.info(f"Loading CommonVoice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(
            self.cv_manifest_dir / f"cv-en_cuts_train.jsonl.gz"
        )

        return CutSet.mux(librispeech_cuts, gigaspeech_cuts, commonvoice_cuts)
