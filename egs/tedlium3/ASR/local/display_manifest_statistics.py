#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
# 						   Mingshuang Luo)
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
    path = "./data/fbank/tedlium_cuts_train.jsonl.gz"
    path = "./data/fbank/tedlium_cuts_dev.jsonl.gz"
    path = "./data/fbank/tedlium_cuts_test.jsonl.gz"

    cuts = load_manifest_lazy(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
## train
Cuts count: 804789
Total duration (hours): 1370.6
Speech duration (hours): 1370.6 (100.0%)
***
Duration statistics (seconds):
mean    6.1
std     3.1
min     0.5
25%     3.7
50%     6.0
75%     8.3
99.5%   14.9
99.9%   16.6
max     33.3

## dev
Cuts count: 507
Total duration (hours): 1.6
Speech duration (hours): 1.6 (100.0%)
***
Duration statistics (seconds):
mean    11.3
std     5.7
min     0.5
25%     7.5
50%     10.6
75%     14.4
99.5%   29.8
99.9%   37.7
max     39.9

## test
Cuts count: 1155
Total duration (hours): 2.6
Speech duration (hours): 2.6 (100.0%)
***
Duration statistics (seconds):
mean    8.2
std     4.3
min     0.3
25%     4.6
50%     8.2
75%     10.9
99.5%   22.1
99.9%   26.7
max     32.5
"""
