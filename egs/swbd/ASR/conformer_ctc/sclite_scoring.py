#!/usr/bin/env python3
# Copyright 2021 Jiayu Du
# Copyright 2022 Johns Hopkins University (Author: Guanbo Wang)
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


import argparse
import os

conversational_filler = [
    "UH",
    "UHH",
    "UM",
    "EH",
    "MM",
    "HM",
    "AH",
    "HUH",
    "HA",
    "ER",
    "OOF",
    "HEE",
    "ACH",
    "EEE",
    "EW",
    "MHM",
    "HUM",
    "AW",
    "OH",
    "HMM",
    "UMM",
]
unk_tags = ["<UNK>", "<unk>"]
switchboard_garbage_utterance_tags = [
    "[LAUGHTER]",
    "[NOISE]",
    "[VOCALIZED-NOISE]",
    "[SILENCE]",
]
non_scoring_words = (
    conversational_filler + unk_tags + switchboard_garbage_utterance_tags
)


def asr_text_post_processing(text: str) -> str:
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove non-scoring words from evaluation
    remaining_words = []
    text_split = text.split()
    word_to_skip = 0
    for idx, word in enumerate(text_split):
        if word_to_skip > 0:
            word_to_skip -= 1
            continue
        if word in non_scoring_words:
            continue
        elif word == "CANCELLED":
            remaining_words.append("CANCELED")
            continue
        elif word == "AIRFLOW":
            remaining_words.append("AIR")
            remaining_words.append("FLOW")
            continue
        elif word == "PHD":
            remaining_words.append("P")
            remaining_words.append("H")
            remaining_words.append("D")
            continue
        elif word == "UCLA":
            remaining_words.append("U")
            remaining_words.append("C")
            remaining_words.append("L")
            remaining_words.append("A")
            continue
        elif word == "ONTO":
            remaining_words.append("ON")
            remaining_words.append("TO")
            continue
        elif word == "DAY":
            try:
                if text_split[idx + 1] == "CARE":
                    remaining_words.append("DAYCARE")
                word_to_skip = 1
            except:
                remaining_words.append(word)
            continue
        remaining_words.append(word)

    return " ".join(remaining_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script evaluates GigaSpeech ASR result via"
        "SCTK's tool sclite"
    )
    parser.add_argument(
        "ref",
        type=str,
        help="sclite's standard transcription(trn) reference file",
    )
    parser.add_argument(
        "hyp",
        type=str,
        help="sclite's standard transcription(trn) hypothesis file",
    )
    parser.add_argument(
        "work_dir",
        type=str,
        help="working dir",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        os.mkdir(args.work_dir)

    REF = os.path.join(args.work_dir, "REF")
    HYP = os.path.join(args.work_dir, "HYP")
    RESULT = os.path.join(args.work_dir, "RESULT")

    for io in [(args.ref, REF), (args.hyp, HYP)]:
        with open(io[0], "r", encoding="utf8") as fi:
            with open(io[1], "w+", encoding="utf8") as fo:
                for line in fi:
                    line = line.strip()
                    if line:
                        cols = line.split()
                        text = asr_text_post_processing(" ".join(cols[0:-1]))
                        uttid_field = cols[-1]
                        print(f"{text} {uttid_field}", file=fo)

    # GigaSpeech's uttid comforms to swb
    os.system(f"sclite -r {REF} trn -h {HYP} trn -i swb | tee {RESULT}")
