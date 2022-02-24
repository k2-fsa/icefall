#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                  Mingshuang Luo)
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

import numpy as np
from lhotse import load_manifest


def describe(cuts) -> None:
    """
    Print a message describing details about the ``CutSet`` - the number of cuts and the
    duration statistics, including the total duration and the percentage of speech segments.

    Example output:
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

    In the above example, we set 15(>14.9) as the maximum duration of training samples.
    """
    durations = np.array([c.duration for c in cuts])
    speech_durations = np.array(
        [s.duration for c in cuts for s in c.trimmed_supervisions]
    )
    total_sum = durations.sum()
    speech_sum = speech_durations.sum()
    print("Cuts count:", len(cuts))
    print(f"Total duration (hours): {total_sum / 3600:.1f}")
    print(
        f"Speech duration (hours): {speech_sum / 3600:.1f} ({speech_sum / total_sum:.1%})"
    )
    print("***")
    print("Duration statistics (seconds):")
    print(f"mean\t{np.mean(durations):.1f}")
    print(f"std\t{np.std(durations):.1f}")
    print(f"min\t{np.min(durations):.1f}")
    print(f"25%\t{np.percentile(durations, 25):.1f}")
    print(f"50%\t{np.median(durations):.1f}")
    print(f"75%\t{np.percentile(durations, 75):.1f}")
    print(f"99.5%\t{np.percentile(durations, 99.5):.1f}")
    print(f"99.9%\t{np.percentile(durations, 99.9):.1f}")
    print(f"max\t{np.max(durations):.1f}")


def main():
    path = "./data/fbank/cuts_train.json.gz"
    # path = "./data/fbank/cuts_dev.json.gz"
    # path = "./data/fbank/cuts_test.json.gz"

    cuts = load_manifest(path)
    describe(cuts)


if __name__ == "__main__":
    main()
