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
        "./data/fbank/tal_csasr_cuts_train_set.jsonl.gz",
        "./data/fbank/tal_csasr_cuts_dev_set.jsonl.gz",
        "./data/fbank/tal_csasr_cuts_test_set.jsonl.gz",
    ]

    for path in paths:
        print(f"Displaying the statistics for {path}")
        cuts = load_manifest(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Displaying the statistics for ./data/fbank/tal_csasr_cuts_train_set.jsonl.gz
Cuts count: 1050000
Total duration (hours): 1679.0
Speech duration (hours): 1679.0 (100.0%)
***
Duration statistics (seconds):
mean    5.8
std     4.1
min     0.3
25%     2.8
50%     4.4
75%     7.3
99%     18.0
99.5%   18.8
99.9%   20.8
max     36.5
Displaying the statistics for ./data/fbank/tal_csasr_cuts_dev_set.jsonl.gz
Cuts count: 5000
Total duration (hours): 8.0
Speech duration (hours): 8.0 (100.0%)
***
Duration statistics (seconds):
mean    5.8
std     4.0
min     0.5
25%     2.8
50%     4.5
75%     7.4
99%     17.0
99.5%   17.7
99.9%   19.5
max     21.5
Displaying the statistics for ./data/fbank/tal_csasr_cuts_test_set.jsonl.gz
Cuts count: 15000
Total duration (hours): 23.6
Speech duration (hours): 23.6 (100.0%)
***
Duration statistics (seconds):
mean    5.7
std     4.0
min     0.5
25%     2.8
50%     4.4
75%     7.2
99%     17.2
99.5%   17.9
99.9%   19.6
max     32.3
"""
