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
        "./data/fbank/aidatatang_cuts_train.jsonl.gz",
        "./data/fbank/aidatatang_cuts_dev.jsonl.gz",
        "./data/fbank/aidatatang_cuts_test.jsonl.gz",
    ]

    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest_lazy(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Starting display the statistics for ./data/fbank/aidatatang_cuts_train.jsonl.gz
Cuts count: 494715
Total duration (hours): 422.6
Speech duration (hours): 422.6 (100.0%)
***
Duration statistics (seconds):
mean    3.1
std     1.2
min     1.0
25%     2.3
50%     2.7
75%     3.5
99%     7.2
99.5%   8.0
99.9%   9.5
max     18.1
Starting display the statistics for ./data/fbank/aidatatang_cuts_dev.jsonl.gz
Cuts count: 24216
Total duration (hours): 20.2
Speech duration (hours): 20.2 (100.0%)
***
Duration statistics (seconds):
mean    3.0
std     1.0
min     1.2
25%     2.3
50%     2.7
75%     3.4
99%     6.7
99.5%   7.3
99.9%   8.8
max     11.3
Starting display the statistics for ./data/fbank/aidatatang_cuts_test.jsonl.gz
Cuts count: 48144
Total duration (hours): 40.2
Speech duration (hours): 40.2 (100.0%)
***
Duration statistics (seconds):
mean    3.0
std     1.1
min     0.9
25%     2.3
50%     2.6
75%     3.4
99%     6.9
99.5%   7.5
99.9%   9.0
max     21.8
"""
