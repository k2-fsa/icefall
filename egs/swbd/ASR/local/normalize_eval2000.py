#!/usr/bin/env python3
# Copyright    2023      (authors: Nagendra Goel https://github.com/ngoel17)
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
import re
from typing import Tuple

from lhotse import SupervisionSegment, SupervisionSet
from lhotse.serialization import load_manifest_lazy_or_eager
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sups")
    parser.add_argument("output_sups")
    return parser.parse_args()


def remove_punctutation_and_other_symbol(text: str) -> str:
    text = text.replace("--", " ")
    text = text.replace("//", " ")
    text = text.replace(".", " ")
    text = text.replace("?", " ")
    text = text.replace("~", " ")
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("&", " ")
    text = text.replace("%", " ")
    text = text.replace("*", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    return text


def eval2000_clean_eform(text: str, eform_count) -> str:
    string_to_remove = []
    piece = text.split('">')
    for i in range(0, len(piece)):
        s = piece[i] + '">'
        res = re.search(r"<contraction e_form(.*?)\">", s)
        if res is not None:
            res_rm = res.group(1)
            string_to_remove.append(res_rm)
    for p in string_to_remove:
        eform_string = p
        text = text.replace(eform_string, " ")
    eform_1 = "<contraction e_form"
    text = text.replace(eform_1, " ")
    eform_2 = '">'
    text = text.replace(eform_2, " ")
    # print("TEXT final: ", text)
    return text


def replace_silphone(text: str) -> str:
    text = text.replace("[/BABY CRYING]", " ")
    text = text.replace("[/CHILD]", " ")
    text = text.replace("[[DISTORTED]]", " ")
    text = text.replace("[/DISTORTION]", " ")
    text = text.replace("[[DRAWN OUT]]", " ")
    text = text.replace("[[DRAWN-OUT]]", " ")
    text = text.replace("[[FAINT]]", " ")
    text = text.replace("[SMACK]", " ")
    text = text.replace("[[MUMBLES]]", " ")
    text = text.replace("[[HIGH PITCHED SQUEAKY VOICE]]", " ")
    text = text.replace("[[IN THE LAUGH]]", "[LAUGHTER]")
    text = text.replace("[[LAST WORD SPOKEN WITH A LAUGH]]", "[LAUGHTER]")
    text = text.replace("[[PART OF FIRST SYLLABLE OF PREVIOUS WORD CUT OFF]]", " ")
    text = text.replace("[[PREVIOUS WORD SPOKEN WITH A LAUGH]]", " ")
    text = text.replace("[[PREVIOUS TWO WORDS SPOKEN WHILE LAUGHING]]", " ")
    text = text.replace("[[PROLONGED]]", " ")
    text = text.replace("[/RUNNING WATER]", " ")
    text = text.replace("[[SAYS LAUGHING]]", "[LAUGHTER]")
    text = text.replace("[[SINGING]]", " ")
    text = text.replace("[[SPOKEN WHILE LAUGHING]]", "[LAUGHTER]")
    text = text.replace("[/STATIC]", " ")
    text = text.replace("['THIRTIETH' DRAWN OUT]", " ")
    text = text.replace("[/VOICES]", " ")
    text = text.replace("[[WHISPERED]]", " ")
    text = text.replace("[DISTORTION]", " ")
    text = text.replace("[DISTORTION, HIGH VOLUME ON WAVES]", " ")
    text = text.replace("[BACKGROUND LAUGHTER]", "[LAUGHTER]")
    text = text.replace("[CHILD'S VOICE]", " ")
    text = text.replace("[CHILD SCREAMS]", " ")
    text = text.replace("[CHILD VOICE]", " ")
    text = text.replace("[CHILD YELLING]", " ")
    text = text.replace("[CHILD SCREAMING]", " ")
    text = text.replace("[CHILD'S VOICE IN BACKGROUND]", " ")
    text = text.replace("[CHANNEL NOISE]", " ")
    text = text.replace("[CHANNEL ECHO]", " ")
    text = text.replace("[ECHO FROM OTHER CHANNEL]", " ")
    text = text.replace("[ECHO OF OTHER CHANNEL]", " ")
    text = text.replace("[CLICK]", " ")
    text = text.replace("[DISTORTED]", " ")
    text = text.replace("[BABY CRYING]", " ")
    text = text.replace("[METALLIC KNOCKING SOUND]", " ")
    text = text.replace("[METALLIC SOUND]", " ")

    text = text.replace("[PHONE JIGGLING]", " ")
    text = text.replace("[BACKGROUND SOUND]", " ")
    text = text.replace("[BACKGROUND VOICE]", " ")
    text = text.replace("[BACKGROUND VOICES]", " ")
    text = text.replace("[BACKGROUND NOISE]", " ")
    text = text.replace("[CAR HORNS IN BACKGROUND]", " ")
    text = text.replace("[CAR HORNS]", " ")
    text = text.replace("[CARNATING]", " ")
    text = text.replace("[CRYING CHILD]", " ")
    text = text.replace("[CHOPPING SOUND]", " ")
    text = text.replace("[BANGING]", " ")
    text = text.replace("[CLICKING NOISE]", " ")
    text = text.replace("[CLATTERING]", " ")
    text = text.replace("[ECHO]", " ")
    text = text.replace("[KNOCK]", " ")
    text = text.replace("[NOISE-GOOD]", "[NOISE]")
    text = text.replace("[RIGHT]", " ")
    text = text.replace("[SOUND]", " ")
    text = text.replace("[SQUEAK]", " ")
    text = text.replace("[STATIC]", " ")
    text = text.replace("[[SAYS WITH HIGH-PITCHED SCREAMING LAUGHTER]]", " ")
    text = text.replace("[UH]", "UH")
    text = text.replace("[MN]", "[VOCALIZED-NOISE]")
    text = text.replace("[VOICES]", " ")
    text = text.replace("[WATER RUNNING]", " ")
    text = text.replace("[SOUND OF TWISTING PHONE CORD]", " ")
    text = text.replace("[SOUND OF SOMETHING FALLING]", " ")
    text = text.replace("[SOUND]", " ")
    text = text.replace("[NOISE OF MOVING PHONE]", " ")
    text = text.replace("[SOUND OF RUNNING WATER]", " ")
    text = text.replace("[CHANNEL]", " ")
    text = text.replace("[SILENCE]", " ")
    text = text.replace("-[W]HERE", "WHERE")
    text = text.replace("Y[OU]I-", "YOU I")
    text = text.replace("-[A]ND", "AND")
    text = text.replace("JU[ST]", "JUST")
    text = text.replace("{BREATH}", " ")
    text = text.replace("{BREATHY}", " ")
    text = text.replace("{CHANNEL NOISE}", " ")
    text = text.replace("{CLEAR THROAT}", " ")

    text = text.replace("{CLEARING THROAT}", " ")
    text = text.replace("{CLEARS THROAT}", " ")
    text = text.replace("{COUGH}", " ")
    text = text.replace("{DRAWN OUT}", " ")
    text = text.replace("{EXHALATION}", " ")
    text = text.replace("{EXHALE}", " ")
    text = text.replace("{GASP}", " ")
    text = text.replace("{HIGH SQUEAL}", " ")
    text = text.replace("{INHALE}", " ")
    text = text.replace("{LAUGH}", "[LAUGHTER]")
    text = text.replace("{LAUGH}", "[LAUGHTER]")
    text = text.replace("{LAUGH}", "[LAUGHTER]")
    text = text.replace("{LIPSMACK}", " ")
    text = text.replace("{LIPSMACK}", " ")

    text = text.replace("{NOISE OF DISGUST}", " ")
    text = text.replace("{SIGH}", " ")
    text = text.replace("{SNIFF}", " ")
    text = text.replace("{SNORT}", " ")
    text = text.replace("{SHARP EXHALATION}", " ")
    text = text.replace("{BREATH LAUGH}", " ")

    text = text.replace("[LAUGHTER]", " ")
    text = text.replace("[NOISE]", " ")
    text = text.replace("[VOCALIZED-NOISE]", " ")
    text = text.replace("-", " ")
    return text


def remove_languagetag(text: str) -> str:
    langtag = re.findall(r"<(.*?)>", text)
    for t in langtag:
        text = text.replace(t, " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    return text


def eval2000_normalizer(text: str) -> str:
    # print("TEXT original: ",text)
    eform_count = text.count("contraction e_form")
    # print("eform corunt:", eform_count)
    if eform_count > 0:
        text = eval2000_clean_eform(text, eform_count)
    text = text.upper()
    text = remove_languagetag(text)
    text = replace_silphone(text)
    text = remove_punctutation_and_other_symbol(text)
    text = text.replace("IGNORE_TIME_SEGMENT_IN_SCORING", " ")
    text = text.replace("IGNORE_TIME_SEGMENT_SCORING", " ")
    spaces = re.findall(r"\s+", text)
    for sp in spaces:
        text = text.replace(sp, " ")
    text = text.strip()
    # text = self.whitespace_regexp.sub(" ", text).strip()
    # print(text)
    return text


def main():
    args = get_args()
    sups = load_manifest_lazy_or_eager(args.input_sups)
    assert isinstance(sups, SupervisionSet)

    tot, skip = 0, 0
    with SupervisionSet.open_writer(args.output_sups) as writer:
        for sup in tqdm(sups, desc="Normalizing supervisions"):
            tot += 1
            sup.text = eval2000_normalizer(sup.text)
            if not sup.text:
                skip += 1
                continue
            writer.write(sup)


if __name__ == "__main__":
    main()
