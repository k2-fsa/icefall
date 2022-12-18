#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
# 						                           Mingshuang Luo)
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
See the function `remove_short_and_long_utt()`
in ../../../librispeech/ASR/transducer/train.py
for usage.
"""


from lhotse import load_manifest_lazy


def main():
    paths = [
        "./data/fbank/cuts_S.jsonl.gz",
        "./data/fbank/cuts_M.jsonl.gz",
        "./data/fbank/cuts_L.jsonl.gz",
        "./data/fbank/cuts_DEV.jsonl.gz",
        "./data/fbank/cuts_TEST_NET.jsonl.gz",
        "./data/fbank/cuts_TEST_MEETING.jsonl.gz",
    ]

    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest_lazy(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Starting display the statistics for ./data/fbank/cuts_L.jsonl.gz

Cuts count: 43874235
Total duration (hours): 30217.3
Speech duration (hours): 30217.3 (100.0%)
***
Duration statistics (seconds):
mean    2.5
std     1.7
min     0.2
25%     1.4
50%     2.0
75%     3.0
99%     8.4
99.5%   9.1
99.9%   15.4
max     405.1

Starting display the statistics for ./data/fbank/cuts_S.jsonl.gz
Duration statistics (seconds):
mean    2.4
std     1.8
min     0.2
25%     1.4
50%     2.0
75%     2.9
99%     8.0
99.5%   8.7
99.9%   11.9
max     405.1

Starting display the statistics for ./data/fbank/cuts_M.jsonl.gz
Cuts count: 4543341
Total duration (hours): 3021.1
Speech duration (hours): 3021.1 (100.0%)
***
Duration statistics (seconds):
mean    2.4
std     1.6
min     0.2
25%     1.4
50%     2.0
75%     2.9
99%     8.0
99.5%   8.8
99.9%   12.1
max     405.1

Starting display the statistics for ./data/fbank/cuts_DEV.jsonl.gz
Cuts count: 13825
Total duration (hours): 20.0
Speech duration (hours): 20.0 (100.0%)
***
Duration statistics (seconds):
mean    5.2
std     2.2
min     1.0
25%     3.3
50%     4.9
75%     7.0
99%     9.6
99.5%   9.8
99.9%   10.0
max     10.0

Starting display the statistics for ./data/fbank/cuts_TEST_NET.jsonl.gz
Cuts count: 24774
Total duration (hours): 23.1
Speech duration (hours): 23.1 (100.0%)
***
Duration statistics (seconds):
mean    3.4
std     2.6
min     0.1
25%     1.4
50%     2.4
75%     4.8
99%     13.1
99.5%   14.5
99.9%   18.5
max     33.3

Starting display the statistics for ./data/fbank/cuts_TEST_MEETING.jsonl.gz
Cuts count: 8370
Total duration (hours): 15.2
Speech duration (hours): 15.2 (100.0%)
***
Duration statistics (seconds):
mean    6.5
std     3.5
min     0.8
25%     3.7
50%     5.8
75%     8.8
99%     15.2
99.5%   16.0
99.9%   18.8
max     24.6

"""
