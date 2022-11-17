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
]
unk_tags = ["<UNK>", "<unk>"]
gigaspeech_punctuations = [
    "<COMMA>",
    "<PERIOD>",
    "<QUESTIONMARK>",
    "<EXCLAMATIONPOINT>",
]
gigaspeech_garbage_utterance_tags = ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"]
non_scoring_words = (
    conversational_filler
    + unk_tags
    + gigaspeech_punctuations
    + gigaspeech_garbage_utterance_tags
)


def asr_text_post_processing(text: str) -> str:
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace("-", " ")

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
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
