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

See the function `remove_short_and_long_utt()` in transducer/train.py
for usage.
"""


from lhotse import load_manifest


def main():
    # path = "./data/fbank/cuts_train.jsonl.gz"
    path = "./data/fbank/cuts_dev.jsonl.gz"
    # path = "./data/fbank/cuts_test.jsonl.gz"

    cuts = load_manifest(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
# train

Cuts count: 1125309
Total duration (hours): 3403.9
Speech duration (hours): 3403.9 (100.0%)
***
Duration statistics (seconds):
mean    10.9
std     10.1
min     0.2
25%     5.2
50%     7.8
75%     12.7
99%     52.0
99.5%   65.1
99.9%   99.5
max     228.9


# test
Cuts count: 5365
Total duration (hours): 9.6
Speech duration (hours): 9.6 (100.0%)
***
Duration statistics (seconds):
mean    6.4
std     1.5
min     1.6
25%     5.3
50%     6.5
75%     7.6
99%     9.5
99.5%   9.7
99.9%   10.3
max     12.4

# dev
Cuts count: 5002
Total duration (hours): 8.5
Speech duration (hours): 8.5 (100.0%)
***
Duration statistics (seconds):
mean    6.1
std     1.7
min     1.5
25%     4.8
50%     6.2
75%     7.4
99%     9.5
99.5%   9.7
99.9%   10.1
max     20.3

"""
