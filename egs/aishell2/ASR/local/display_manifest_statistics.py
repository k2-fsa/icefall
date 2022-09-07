#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

"""
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in transducer_stateless/train.py
for usage.
"""


from lhotse import load_manifest_lazy


def main():
    paths = [
        "./data/fbank/aishell2_cuts_train.jsonl.gz",
        "./data/fbank/aishell2_cuts_dev.jsonl.gz",
        "./data/fbank/aishell2_cuts_test.jsonl.gz",
    ]

    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest_lazy(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Starting display the statistics for ./data/fbank/aishell2_cuts_train.jsonl.gz
Cuts count: 3026106
Total duration (hours): 3021.2
Speech duration (hours): 3021.2 (100.0%)
***
Duration statistics (seconds):
mean	3.6
std	1.5
min	0.3
25%	2.4
50%	3.3
75%	4.4
99%	8.2
99.5%	8.9
99.9%	10.6
max	21.5
Starting display the statistics for ./data/fbank/aishell2_cuts_dev.jsonl.gz
Cuts count: 2500
Total duration (hours): 2.0
Speech duration (hours): 2.0 (100.0%)
***
Duration statistics (seconds):
mean	2.9
std	1.0
min	1.1
25%	2.2
50%	2.7
75%	3.4
99%	6.3
99.5%	6.7
99.9%	7.8
max	9.4
Starting display the statistics for ./data/fbank/aishell2_cuts_test.jsonl.gz
Cuts count: 5000
Total duration (hours): 4.0
Speech duration (hours): 4.0 (100.0%)
***
Duration statistics (seconds):
mean	2.9
std	1.0
min	1.1
25%	2.2
50%	2.7
75%	3.3
99%	6.2
99.5%	6.6
99.9%	7.7
max	8.5
"""
