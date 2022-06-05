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


from lhotse import load_manifest


def main():
    paths = [
        "./data/fbank/cuts_train_S.json.gz",
        "./data/fbank/cuts_train_M.json.gz",
        "./data/fbank/cuts_train_L.json.gz",
        "./data/fbank/cuts_test.json.gz",
    ]

    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Starting display the statistics for ./data/fbank/cuts_train_S.json.gz
Cuts count: 91995
Total duration (hours): 95.8
Speech duration (hours): 95.8 (100.0%)
***
Duration statistics (seconds):
mean    3.7
std     7.1
min     0.1
25%     0.9
50%     2.5
75%     5.4
99%     15.3
99.5%   17.5
99.9%   23.3
max     1021.7
Starting display the statistics for ./data/fbank/cuts_train_M.json.gz
Cuts count: 177195
Total duration (hours): 179.5
Speech duration (hours): 179.5 (100.0%)
***
Duration statistics (seconds):
mean    3.6
std     6.4
min     0.0
25%     0.9
50%     2.4
75%     5.2
99%     14.9
99.5%   17.0
99.9%   23.5
max     990.4
Starting display the statistics for ./data/fbank/cuts_train_L.json.gz
Cuts count: 37572
Total duration (hours): 49.1
Speech duration (hours): 49.1 (100.0%)
***
Duration statistics (seconds):
mean    4.7
std     4.0
min     0.2
25%     1.6
50%     3.7
75%     6.7
99%     17.5
99.5%   19.8
99.9%   26.2
max     87.4
Starting display the statistics for ./data/fbank/cuts_test.json.gz
Cuts count: 10574
Total duration (hours): 12.1
Speech duration (hours): 12.1 (100.0%)
***
Duration statistics (seconds):
mean    4.1
std     3.4
min     0.2
25%     1.4
50%     3.2
75%     5.8
99%     14.4
99.5%   14.9
99.9%   16.5
max     17.9
"""
