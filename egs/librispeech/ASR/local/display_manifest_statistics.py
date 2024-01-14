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


from lhotse import load_manifest_lazy


def main():
    #  path = "./data/fbank/librispeech_cuts_train-clean-100.jsonl.gz"
    #  path = "./data/fbank/librispeech_cuts_train-clean-360.jsonl.gz"
    #  path = "./data/fbank/librispeech_cuts_train-other-500.jsonl.gz"
    #  path = "./data/fbank/librispeech_cuts_dev-clean.jsonl.gz"
    #  path = "./data/fbank/librispeech_cuts_dev-other.jsonl.gz"
    #  path = "./data/fbank/librispeech_cuts_test-clean.jsonl.gz"
    path = "./data/fbank/librispeech_cuts_test-other.jsonl.gz"

    cuts = load_manifest_lazy(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
## train-clean-100
Cuts count: 85617
Total duration (hours): 303.8
Speech duration (hours): 303.8 (100.0%)
***
Duration statistics (seconds):
mean    12.8
std     3.8
min     1.3
0.1%    1.9
0.5%    2.2
1%      2.5
5%      4.2
10%     6.4
25%     11.4
50%     13.8
75%     15.3
90%     16.7
95%     17.3
99%     18.1
99.5%   18.4
99.9%   18.8
max     27.2

## train-clean-360
Cuts count: 312042
Total duration (hours): 1098.2
Speech duration (hours): 1098.2 (100.0%)
***
Duration statistics (seconds):
mean    12.7
std     3.8
min     1.0
0.1%    1.8
0.5%    2.2
1%      2.5
5%      4.2
10%     6.2
25%     11.2
50%     13.7
75%     15.3
90%     16.6
95%     17.3
99%     18.1
99.5%   18.4
99.9%   18.8
max     33.0

## train-other 500
Cuts count: 446064
Total duration (hours): 1500.6
Speech duration (hours): 1500.6 (100.0%)
***
Duration statistics (seconds):
mean    12.1
std     4.2
min     0.8
0.1%    1.7
0.5%    2.1
1%      2.3
5%      3.5
10%     5.0
25%     9.8
50%     13.4
75%     15.1
90%     16.5
95%     17.2
99%     18.1
99.5%   18.4
99.9%   18.9
max     31.0

## dev-clean
Cuts count: 2703
Total duration (hours): 5.4
Speech duration (hours): 5.4 (100.0%)
***
Duration statistics (seconds):
mean    7.2
std     4.7
min     1.4
0.1%    1.6
0.5%    1.8
1%      1.9
5%      2.4
10%     2.7
25%     3.8
50%     5.9
75%     9.3
90%     13.3
95%     16.4
99%     23.8
99.5%   28.5
99.9%   32.3
max     32.6

## dev-other
Cuts count: 2864
Total duration (hours): 5.1
Speech duration (hours): 5.1 (100.0%)
***
Duration statistics (seconds):
mean    6.4
std     4.3
min     1.1
0.1%    1.3
0.5%    1.7
1%      1.8
5%      2.2
10%     2.6
25%     3.5
50%     5.3
75%     7.9
90%     12.0
95%     15.0
99%     22.2
99.5%   27.1
99.9%   32.4
max     35.2

## test-clean
Cuts count: 2620
Total duration (hours): 5.4
Speech duration (hours): 5.4 (100.0%)
***
Duration statistics (seconds):
mean    7.4
std     5.2
min     1.3
0.1%    1.6
0.5%    1.8
1%      2.0
5%      2.3
10%     2.7
25%     3.7
50%     5.8
75%     9.6
90%     14.6
95%     17.8
99%     25.5
99.5%   28.4
99.9%   32.8
max     35.0

## test-other
Cuts count: 2939
Total duration (hours): 5.3
Speech duration (hours): 5.3 (100.0%)
***
Duration statistics (seconds):
mean    6.5
std     4.4
min     1.2
0.1%    1.5
0.5%    1.8
1%      1.9
5%      2.3
10%     2.6
25%     3.4
50%     5.2
75%     8.2
90%     12.6
95%     15.8
99%     21.4
99.5%   23.8
99.9%   33.5
max     34.5
"""
