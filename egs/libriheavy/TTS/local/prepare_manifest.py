#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
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

import gzip
import json
import re
import sys
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from tn.english.normalizer import Normalizer as EnNormalizer
from tqdm import tqdm

from icefall.utils import str2bool


class TextNormlizer:
    def __init__(self):
        self.en_tn_model = EnNormalizer(cache_dir="/tmp/tn", overwrite_cache=False)

    def __call__(self, cut):
        text = cut["supervisions"][0]["custom"]["texts"][0]

        # Process brackets
        text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
        text = re.sub(r"\([^\)]*\)", " ", text)

        # Apply mappings
        table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
        text = text.translate(table)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        text = self.en_tn_model.normalize(text)

        cut["supervisions"][0]["text"] = text
        del cut["supervisions"][0]["custom"]
        del cut["custom"]

        return cut


# Assign text of the supervisions and remove unnecessary entries.
def main():
    assert len(sys.argv) == 3, "Usage: ./local/prepare_manifest.py INPUT OUTPUT_DIR"
    fname = Path(sys.argv[1]).name
    oname = Path(sys.argv[2]) / fname

    tn = TextNormlizer()
    with gzip.open(sys.argv[1], "r") as fin, ProcessPoolExecutor() as ex:
        futures = []
        cuts = []
        for line in tqdm(fin, desc="Distributing tasks"):
            cut = json.loads(line)
            futures.append(ex.submit(tn, cut))

    with gzip.open(oname, "w") as fout:
        for future in tqdm(futures, desc="Processing"):
            cut = future.result()
            fout.write((json.dumps(cut) + "\n").encode())


if __name__ == "__main__":
    main()
