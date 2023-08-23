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
    #  path = "./data/fbank/swbd_cuts_rt03.jsonl.gz"
    path = "./data/fbank/eval2000/eval2000_cuts_all.jsonl.gz"
    # path = "./data/fbank/swbd_cuts_all.jsonl.gz"

    cuts = load_manifest_lazy(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
Training Cut statistics:
╒═══════════════════════════╤═══════════╕
│ Cuts count:               │ 167244    │
├───────────────────────────┼───────────┤
│ Total duration (hh:mm:ss) │ 281:01:26 │
├───────────────────────────┼───────────┤
│ mean                      │ 6.0       │
├───────────────────────────┼───────────┤
│ std                       │ 3.3       │
├───────────────────────────┼───────────┤
│ min                       │ 2.0       │
├───────────────────────────┼───────────┤
│ 25%                       │ 3.2       │
├───────────────────────────┼───────────┤
│ 50%                       │ 5.2       │
├───────────────────────────┼───────────┤
│ 75%                       │ 8.3       │
├───────────────────────────┼───────────┤
│ 99%                       │ 14.4      │
├───────────────────────────┼───────────┤
│ 99.5%                     │ 14.7      │
├───────────────────────────┼───────────┤
│ 99.9%                     │ 15.0      │
├───────────────────────────┼───────────┤
│ max                       │ 57.5      │
├───────────────────────────┼───────────┤
│ Recordings available:     │ 167244    │
├───────────────────────────┼───────────┤
│ Features available:       │ 167244    │
├───────────────────────────┼───────────┤
│ Supervisions available:   │ 167244    │
╘═══════════════════════════╧═══════════╛
Speech duration statistics:
╒══════════════════════════════╤═══════════╤══════════════════════╕
│ Total speech duration        │ 281:01:26 │ 100.00% of recording │
├──────────────────────────────┼───────────┼──────────────────────┤
│ Total speaking time duration │ 281:01:26 │ 100.00% of recording │
├──────────────────────────────┼───────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00  │ 0.00% of recording   │
╘══════════════════════════════╧═══════════╧══════════════════════╛

Eval2000 Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 2709     │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 01:39:19 │
├───────────────────────────┼──────────┤
│ mean                      │ 2.2      │
├───────────────────────────┼──────────┤
│ std                       │ 1.8      │
├───────────────────────────┼──────────┤
│ min                       │ 0.1      │
├───────────────────────────┼──────────┤
│ 25%                       │ 0.7      │
├───────────────────────────┼──────────┤
│ 50%                       │ 1.7      │
├───────────────────────────┼──────────┤
│ 75%                       │ 3.1      │
├───────────────────────────┼──────────┤
│ 99%                       │ 8.0      │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 8.3      │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 11.3     │
├───────────────────────────┼──────────┤
│ max                       │ 14.1     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 2709     │
├───────────────────────────┼──────────┤
│ Features available:       │ 0        │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 2709     │
╘═══════════════════════════╧══════════╛
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 01:39:19 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 01:39:19 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛
"""
