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
import sys
from pathlib import Path

from icefall.utils import str2bool


def simple_cleanup(text: str) -> str:
    table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
    text = text.translate(table)
    return text.strip()


# Assign text of the supervisions and remove unnecessary entries.
def main():
    assert (
        len(sys.argv) == 4
    ), "Usage: ./local/prepare_manifest.py INPUT OUTPUT_DIR KEEP_CUSTOM_FIELDS"
    fname = Path(sys.argv[1]).name
    oname = Path(sys.argv[2]) / fname
    keep_custom_fields = str2bool(sys.argv[3])
    with gzip.open(sys.argv[1], "r") as fin, gzip.open(oname, "w") as fout:
        for line in fin:
            cut = json.loads(line)
            cut["supervisions"][0]["text"] = simple_cleanup(
                cut["supervisions"][0]["custom"]["texts"][0]
            )
            if not keep_custom_fields:
                del cut["supervisions"][0]["custom"]
                del cut["custom"]
            fout.write((json.dumps(cut) + "\n").encode())


if __name__ == "__main__":
    main()
