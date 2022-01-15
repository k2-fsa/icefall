#!/usr/bin/env python3

import argparse
import re
from typing import Tuple

from tqdm import tqdm

from lhotse import SupervisionSet, SupervisionSegment
from lhotse.serialization import load_manifest_lazy_or_eager


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sups")
    parser.add_argument("output_sups")
    return parser.parse_args()


class FisherSwbdNormalizer:
    """
    Note: the functions "normalize" and "keep" implement the logic similar to
    Kaldi's data prep scripts for Fisher:
      https://github.com/kaldi-asr/kaldi/blob/master/egs/fisher_swbd/s5/local/fisher_data_prep.sh
    and for SWBD:
      https://github.com/kaldi-asr/kaldi/blob/master/egs/fisher_swbd/s5/local/swbd1_data_prep.sh

    One notable difference is that we don't change [cough], [lipsmack], etc. to [noise]. 
    We also don't implement all the edge cases of normalization from Kaldi 
    (hopefully won't make too much difference).
    """


    def __init__(self) -> None:

        self.remove_regexp_before = re.compile(
            r"|".join([
                # special symbols
                r"\[\[SKIP.*\]\]",
                r"\[SKIP.*\]",
                r"\[PAUSE.*\]",
                r"\[SILENCE\]",
                r"<B_ASIDE>",
                r"<E_ASIDE>",
            ])
        )

        # tuples of (pattern, replacement)
        # note: Kaldi replaces sighs, coughs, etc with [noise].
        #       We don't do that here.
        #       We also uppercase the text as the first operation.
        self.replace_regexps: Tuple[re.Pattern, str] = [
            # SWBD: 
            # [LAUGHTER-STORY] -> STORY
            (re.compile(r"\[LAUGHTER-(.*?)\]"), r"\1"),
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
            (re.compile(r"\[LAUGH.*?\]"), r"[LAUGHTER]"),
            (re.compile(r"\[SIGH.*?\]"), r"[SIGH]"),
            (re.compile(r"\[COUGH.*?\]"), r"[COUGH]"),
            (re.compile(r"\[MN.*?\]"), r"[VOCALIZED-NOISE]"),
            (re.compile(r"\[BREATH.*?\]"), r"[BREATH]"),
            (re.compile(r"\[LIPSMACK.*?\]"), r"[LIPSMACK]"),
            (re.compile(r"\[SNEEZE.*?\]"), r"[SNEEZE]"),
            # abbreviations
            (re.compile(r"(\w)\.(\w)\.(\w)",), r"\1 \2 \3"),
            (re.compile(r"(\w)\.(\w)",), r"\1 \2"),
            (re.compile(r"\._",), r" "),
            (re.compile(r"_(\w)",), r"\1"),
            (re.compile(r"(\w)\.s",), r"\1's"),
            # words between apostrophes
            (re.compile(r"'(\S*?)'"), r"\1"),
            # dangling dashes (2 passes)
            (re.compile(r"\s-\s"), r" "),
            (re.compile(r"\s-\s"), r" "),
            # special symbol with trailing dash
            (re.compile(r"(\[.*?\])-"), r"\1"),
        ]

        # unwanted symbols in the transcripts
        self.remove_regexp_after = re.compile(
            r"|".join([
                # remaining punctuation
                r"\.",
                r",",
                r"\?",
                r"{",
                r"}",
                r"~",
                r"_\d",
            ])
        )

        self.whitespace_regexp = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        text = text.upper()

        # first remove
        text = self.remove_regexp_before.sub("", text)

        # then replace
        for pattern, sub in self.replace_regexps:
            text = pattern.sub(sub, text)

        # then remove
        text = self.remove_regexp_after.sub("", text)

        # then clean up whitespace
        text = self.whitespace_regexp.sub(" ", text).strip()

        return text


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

    normalizer = Normalizer()

    tot, skip = 0, 0
    with SupervisionSet.open_writer(args.output_sups) as writer:
        for sup in tqdm(sups, desc="Normalizing supervisions"):
            tot += 1

            if not keep(sup):
                skip += 1
                continue

            sup.text = normalizer.normalize(sup.text)
            if not sup.text:
                skip += 1
                continue

            writer.write(sup)


def test():
    normalizer = Normalizer()
    for text in [
        "[laughterr]",
        "[laugh] oh this is great [silence] <B_ASIDE> yes",
        "[laugh] oh this is [laught] this is great [silence] <B_ASIDE> yes",
        "i don't kn- - know a.b.c's",
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
    ]:
        print(text)
        print(normalizer.normalize(text))
        print()

if __name__ == "__main__":
    # test()
    main()
