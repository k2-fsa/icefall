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
    #  path = "./data/fbank/aishell_cuts_train.jsonl.gz"
    #  path = "./data/fbank/aishell_cuts_test.jsonl.gz"
    path = "./data/fbank/aishell_cuts_dev.jsonl.gz"
    #  path = "./data/fbank/aidatatang_cuts_train.jsonl.gz"
    #  path = "./data/fbank/aidatatang_cuts_test.jsonl.gz"
    #  path = "./data/fbank/aidatatang_cuts_dev.jsonl.gz"

    cuts = load_manifest_lazy(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
## train (after speed perturb)
Cuts count: 360294
Total duration (hours): 455.6
Speech duration (hours): 455.6 (100.0%)
***
Duration statistics (seconds):
mean    4.6
std     1.4
min     1.1
0.1%    1.8
0.5%    2.2
1%      2.3
5%      2.7
10%     3.0
10%     3.0
25%     3.5
50%     4.3
75%     5.4
90%     6.5
95%     7.2
99%     8.8
99.5%   9.4
99.9%   10.9
max     16.1

## test
Cuts count: 7176
Total duration (hours): 10.0
Speech duration (hours): 10.0 (100.0%)
***
Duration statistics (seconds):
mean    5.0
std     1.6
min     1.9
0.1%    2.2
0.5%    2.4
1%      2.6
5%      3.0
10%     3.2
10%     3.2
25%     3.8
50%     4.7
75%     5.9
90%     7.3
95%     8.2
99%     9.9
99.5%   10.7
99.9%   11.9
max     14.7

## dev
Cuts count: 14326
Total duration (hours): 18.1
Speech duration (hours): 18.1 (100.0%)
***
Duration statistics (seconds):
mean    4.5
std     1.3
min     1.6
0.1%    2.1
0.5%    2.3
1%      2.4
5%      2.9
10%     3.1
10%     3.1
25%     3.5
50%     4.3
75%     5.4
90%     6.4
95%     7.0
99%     8.4
99.5%   8.9
99.9%   10.3
max     12.5

## aidatatang_200zh (train)
Cuts count: 164905
Total duration (hours): 139.9
Speech duration (hours): 139.9 (100.0%)
***
Duration statistics (seconds):
mean    3.1
std     1.1
min     1.1
0.1%    1.5
0.5%    1.7
1%      1.8
5%      2.0
10%     2.1
10%     2.1
25%     2.3
50%     2.7
75%     3.4
90%     4.6
95%     5.4
99%     7.1
99.5%   7.8
99.9%   9.1
max     16.3

## aidatatang_200zh (test)
Cuts count: 48144
Total duration (hours): 40.2
Speech duration (hours): 40.2 (100.0%)
***
Duration statistics (seconds):
mean    3.0
std     1.1
min     0.9
0.1%    1.5
0.5%    1.8
1%      1.8
5%      2.0
10%     2.1
10%     2.1
25%     2.3
50%     2.6
75%     3.4
90%     4.4
95%     5.2
99%     6.9
99.5%   7.5
99.9%   9.0
max     21.8

## aidatatang_200zh (dev)
Cuts count: 24216
Total duration (hours): 20.2
Speech duration (hours): 20.2 (100.0%)
***
Duration statistics (seconds):
mean    3.0
std     1.0
min     1.2
0.1%    1.6
0.5%    1.7
1%      1.8
5%      2.0
10%     2.1
10%     2.1
25%     2.3
50%     2.7
75%     3.4
90%     4.4
95%     5.1
99%     6.7
99.5%   7.3
99.9%   8.8
max     11.3
"""
