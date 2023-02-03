#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  The University of Electro-Communications (author: Teo Wen Shen)  # noqa
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

import argparse
from pathlib import Path

from lhotse import CutSet, load_manifest

ARGPARSE_DESCRIPTION = """
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in
pruned_transducer_stateless5/train.py for usage.
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description=ARGPARSE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--manifest-dir", type=Path, help="Path to cutset manifests")

    return parser.parse_args()


def main():
    args = get_parser()

    for path in args.manifest_dir.glob("csj_cuts_*.jsonl.gz"):

        cuts: CutSet = load_manifest(path)

        print("\n---------------------------------\n")
        print(path.name + ":")
        cuts.describe()


if __name__ == "__main__":
    main()

"""
## eval1
Cuts count: 1272
Total duration (hh:mm:ss): 01:50:07
Speech duration (hh:mm:ss): 01:50:07 (100.0%)
Duration statistics (seconds):
mean	5.2
std	3.9
min	0.2
25%	1.9
50%	4.0
75%	8.1
99%	14.3
99.5%	14.7
99.9%	16.0
max	16.9
Recordings available: 1272
Features available: 1272
Supervisions available: 1272
SUPERVISION custom fields:
- fluent (in 1272 cuts)
- disfluent (in 1272 cuts)
- number (in 1272 cuts)
- symbol (in 1272 cuts)

## eval2
Cuts count: 1292
Total duration (hh:mm:ss): 01:56:50
Speech duration (hh:mm:ss): 01:56:50 (100.0%)
Duration statistics (seconds):
mean	5.4
std	3.9
min	0.1
25%	2.1
50%	4.6
75%	8.6
99%	14.1
99.5%	15.2
99.9%	16.1
max	16.9
Recordings available: 1292
Features available: 1292
Supervisions available: 1292
SUPERVISION custom fields:
- fluent (in 1292 cuts)
- number (in 1292 cuts)
- symbol (in 1292 cuts)
- disfluent (in 1292 cuts)

## eval3
Cuts count: 1385
Total duration (hh:mm:ss): 01:19:21
Speech duration (hh:mm:ss): 01:19:21 (100.0%)
Duration statistics (seconds):
mean	3.4
std	3.0
min	0.2
25%	1.2
50%	2.5
75%	4.6
99%	12.7
99.5%	13.7
99.9%	15.0
max	15.9
Recordings available: 1385
Features available: 1385
Supervisions available: 1385
SUPERVISION custom fields:
- number (in 1385 cuts)
- symbol (in 1385 cuts)
- fluent (in 1385 cuts)
- disfluent (in 1385 cuts)

## valid
Cuts count: 4000
Total duration (hh:mm:ss): 05:08:09
Speech duration (hh:mm:ss): 05:08:09 (100.0%)
Duration statistics (seconds):
mean	4.6
std	3.8
min	0.1
25%	1.5
50%	3.4
75%	7.0
99%	13.8
99.5%	14.8
99.9%	16.0
max	17.3
Recordings available: 4000
Features available: 4000
Supervisions available: 4000
SUPERVISION custom fields:
- fluent (in 4000 cuts)
- symbol (in 4000 cuts)
- disfluent (in 4000 cuts)
- number (in 4000 cuts)

## train
Cuts count: 1291134
Total duration (hh:mm:ss): 1596:37:27
Speech duration (hh:mm:ss): 1596:37:27 (100.0%)
Duration statistics (seconds):
mean	4.5
std	3.6
min	0.0
25%	1.6
50%	3.3
75%	6.4
99%	14.0
99.5%	14.8
99.9%	16.6
max	27.8
Recordings available: 1291134
Features available: 1291134
Supervisions available: 1291134
SUPERVISION custom fields:
- disfluent (in 1291134 cuts)
- fluent (in 1291134 cuts)
- symbol (in 1291134 cuts)
- number (in 1291134 cuts)
"""
