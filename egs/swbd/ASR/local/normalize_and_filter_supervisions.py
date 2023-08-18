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


# replacement function to convert lowercase letter to uppercase
def to_upper(match_obj):
    if match_obj.group() is not None:
        return match_obj.group().upper()


def insert_groups_and_capitalize_3(match):
    return f"{match.group(1)} {match.group(2)} {match.group(3)}".upper()


def insert_groups_and_capitalize_2(match):
    return f"{match.group(1)} {match.group(2)}".upper()


def insert_groups_and_capitalize_1(match):
    return f"{match.group(1)}".upper()


def insert_groups_and_capitalize_1s(match):
    return f"{match.group(1)}".upper() + "'s"


class FisherSwbdNormalizer:
    """Note: the functions "normalize" and "keep" implement the logic
    similar to Kaldi's data prep scripts for Fisher and SWBD: One
    notable difference is that we don't change [cough], [lipsmack],
    etc. to [noise].  We also don't implement all the edge cases of
    normalization from Kaldi (hopefully won't make too much
    difference).
    """

    def __init__(self) -> None:
        self.remove_regexp_before = re.compile(
            r"|".join(
                [
                    # special symbols
                    r"\[\[skip.*\]\]",
                    r"\[skip.*\]",
                    r"\[pause.*\]",
                    r"\[silence\]",
                    r"<b_aside>",
                    r"<e_aside>",
                    r"_1",
                ]
            )
        )

        # tuples of (pattern, replacement)
        # note: Kaldi replaces sighs, coughs, etc with [noise].
        #       We don't do that here.
        #       We also lowercase the text as the first operation.
        self.replace_regexps: Tuple[re.Pattern, str] = [
            # SWBD:
            # [LAUGHTER-STORY] -> STORY
            (re.compile(r"\[laughter-(.*?)\]"), r"\1"),
            # [WEA[SONABLE]-/REASONABLE]
            (re.compile(r"\[\S+/(\S+)\]"), r"\1"),
            # -[ADV]AN[TAGE]- -> AN
            (re.compile(r"-?\[.*?\](\w+)\[.*?\]-?"), r"\1-"),
            # ABSOLUTE[LY]- -> ABSOLUTE-
            (re.compile(r"(\w+)\[.*?\]-?"), r"\1-"),
            # [AN]Y- -> Y-
            # -[AN]Y- -> Y-
            (re.compile(r"-?\[.*?\](\w+)-?"), r"\1-"),
            # special tokens
            (re.compile(r"\[laugh.*?\]"), r"[laughter]"),
            (re.compile(r"\[sigh.*?\]"), r"[sigh]"),
            (re.compile(r"\[cough.*?\]"), r"[cough]"),
            (re.compile(r"\[mn.*?\]"), r"[vocalized-noise]"),
            (re.compile(r"\[breath.*?\]"), r"[breath]"),
            (re.compile(r"\[lipsmack.*?\]"), r"[lipsmack]"),
            (re.compile(r"\[sneeze.*?\]"), r"[sneeze]"),
            # abbreviations
            (
                re.compile(
                    r"(\w)\.(\w)\.(\w)",
                ),
                insert_groups_and_capitalize_3,
            ),
            (
                re.compile(
                    r"(\w)\.(\w)",
                ),
                insert_groups_and_capitalize_2,
            ),
            (
                re.compile(
                    r"([a-h,j-z])\.",
                ),
                insert_groups_and_capitalize_1,
            ),
            (
                re.compile(
                    r"\._",
                ),
                r" ",
            ),
            (
                re.compile(
                    r"_(\w)",
                ),
                insert_groups_and_capitalize_1,
            ),
            (
                re.compile(
                    r"(\w)\.s",
                ),
                insert_groups_and_capitalize_1s,
            ),
            (
                re.compile(
                    r"([A-Z])\'s",
                ),
                insert_groups_and_capitalize_1s,
            ),
            (
                re.compile(
                    r"(\s\w\b|^\w\b)",
                ),
                insert_groups_and_capitalize_1,
            ),
            # words between apostrophes
            (re.compile(r"'(\S*?)'"), r"\1"),
            # dangling dashes (2 passes)
            (re.compile(r"\s-\s"), r" "),
            (re.compile(r"\s-\s"), r" "),
            # special symbol with trailing dash
            (re.compile(r"(\[.*?\])-"), r"\1"),
            # Just remove all dashes
            (re.compile(r"-"), r" "),
        ]

        # unwanted symbols in the transcripts
        self.remove_regexp_after = re.compile(
            r"|".join(
                [
                    # remaining punctuation
                    r"\.",
                    r",",
                    r"\?",
                    r"{",
                    r"}",
                    r"~",
                    r"_\d",
                ]
            )
        )

        self.post_fixes = [
            # Fix an issue related to [VOCALIZED NOISE] after dash removal
            (re.compile(r"\[vocalized noise\]"), "[vocalized-noise]"),
        ]

        self.whitespace_regexp = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        text = text.lower()

        # first remove
        text = self.remove_regexp_before.sub("", text)

        # then replace
        for pattern, sub in self.replace_regexps:
            text = pattern.sub(sub, text)

        # then remove
        text = self.remove_regexp_after.sub("", text)

        # post fixes
        for pattern, sub in self.post_fixes:
            text = pattern.sub(sub, text)

        # then clean up whitespace
        text = self.whitespace_regexp.sub(" ", text).strip()

        return text.upper()


def keep(sup: SupervisionSegment) -> bool:
    if "((" in sup.text:
        return False

    if "<german" in sup.text:
        return False

    return True


def main():
    args = get_args()
    sups = load_manifest_lazy_or_eager(args.input_sups)
    assert isinstance(sups, SupervisionSet)

    normalizer = FisherSwbdNormalizer()

    tot, skip = 0, 0
    with SupervisionSet.open_writer(args.output_sups) as writer:
        for sup in tqdm(sups, desc="Normalizing supervisions"):
            tot += 1

            if not keep(sup):
                skip += 1
                continue

            sup.text = normalizer.normalize(sup.text).upper()
            if not sup.text:
                skip += 1
                continue

            writer.write(sup)
    print(f"tot: {tot}, skip: {skip}")


def test():
    normalizer = FisherSwbdNormalizer()
    for text in [
        "[laughterr] [SILENCE]",
        "[laugh] oh this is great [silence] <B_ASIDE> yes",
        "[laugh] oh this is [laught] this is great [silence] <B_ASIDE> yes",
        "i don't kn- - know A.B.C's",
        "so x. corp is good?",
        "'absolutely yes",
        "absolutely' yes",
        "'absolutely' yes",
        "'absolutely' yes 'aight",
        "ABSOLUTE[LY]",
        "ABSOLUTE[LY]-",
        "[AN]Y",
        "[AN]Y-",
        "[ADV]AN[TAGE]",
        "[ADV]AN[TAGE]-",
        "-[ADV]AN[TAGE]",
        "-[ADV]AN[TAGE]-",
        "[WEA[SONABLE]-/REASONABLE]",
        "[VOCALIZED-NOISE]-",
        "~BULL",
        "Frank E Peretti P E R E T T I",
        "yeah yeah like Double O Seven he's supposed to do it",
        "P A P E R paper",
        "[noise] okay_1 um let me see [laughter] i've been sitting here awhile",
    ]:
        print(text)
        print(normalizer.normalize(text))
        print()


if __name__ == "__main__":
    test()
    # exit()
    main()
