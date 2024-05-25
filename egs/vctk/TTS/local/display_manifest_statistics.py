#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                  Zengrui Jin,)
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

See the function `remove_short_and_long_utt()` in vits/train.py
for usage.
"""


from lhotse import load_manifest_lazy


def main():
    path = "./data/spectrogram/vctk_cuts_all.jsonl.gz"
    cuts = load_manifest_lazy(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 43873    │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 41:02:18 │
├───────────────────────────┼──────────┤
│ mean                      │ 3.4      │
├───────────────────────────┼──────────┤
│ std                       │ 1.2      │
├───────────────────────────┼──────────┤
│ min                       │ 1.2      │
├───────────────────────────┼──────────┤
│ 25%                       │ 2.6      │
├───────────────────────────┼──────────┤
│ 50%                       │ 3.1      │
├───────────────────────────┼──────────┤
│ 75%                       │ 3.8      │
├───────────────────────────┼──────────┤
│ 99%                       │ 8.0      │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 9.1      │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 12.1     │
├───────────────────────────┼──────────┤
│ max                       │ 16.6     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 43873    │
├───────────────────────────┼──────────┤
│ Features available:       │ 43873    │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 43873    │
╘═══════════════════════════╧══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 41:02:18 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 41:02:18 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:01 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛
"""
