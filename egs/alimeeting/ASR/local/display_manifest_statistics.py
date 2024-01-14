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
        "./data/fbank/alimeeting_cuts_train.jsonl.gz",
        "./data/fbank/alimeeting_cuts_eval.jsonl.gz",
        "./data/fbank/alimeeting_cuts_test.jsonl.gz",
    ]

    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest_lazy(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Starting display the statistics for ./data/fbank/alimeeting_cuts_train.jsonl.gz
Cuts count: 559092
Total duration (hours): 424.6
Speech duration (hours): 424.6 (100.0%)
***
Duration statistics (seconds):
mean    2.7
std     3.0
min     0.0
25%     0.7
50%     1.7
75%     3.6
99%     13.6
99.5%   14.7
99.9%   16.2
max     284.3
Starting display the statistics for ./data/fbank/alimeeting_cuts_eval.jsonl.gz
Cuts count: 6457
Total duration (hours): 4.9
Speech duration (hours): 4.9 (100.0%)
***
Duration statistics (seconds):
mean    2.7
std     3.1
min     0.1
25%     0.6
50%     1.6
75%     3.5
99%     13.6
99.5%   14.1
99.9%   14.7
max     15.8
Starting display the statistics for ./data/fbank/alimeeting_cuts_test.jsonl.gz
Cuts count: 16358
Total duration (hours): 12.5
Speech duration (hours): 12.5 (100.0%)
***
Duration statistics (seconds):
mean    2.7
std     2.9
min     0.1
25%     0.7
50%     1.7
75%     3.5
99%     13.7
99.5%   14.2
99.9%   14.8
max     15.7
"""
