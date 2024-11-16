#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Yifan Yang)
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

import gzip
import json
import re
import sys
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from tn.english.normalizer import Normalizer as EnNormalizer
from tqdm import tqdm

from icefall.utils import str2bool


units = [
    "second",
    "minute",
    "quarter",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "decade",
    "century",
    "millisecond",
    "microsecond",
    "nanosecond",
    "am",
    "a.m.",
    "pm",
    "p.m.",
    "a.d.",
    "a.d.",
    "b.c.",
    "bc",
    "monday",
    "mon",
    "tuesday",
    "tue",
    "wednesday",
    "wed",
    "thursday",
    "thu",
    "friday",
    "fri",
    "saturday",
    "sat",
    "sunday",
    "sun",
    "january",
    "jan",
    "february",
    "feb",
    "march",
    "mar",
    "april",
    "apr",
    "may",
    "jun",
    "june",
    "july",
    "jul",
    "august",
    "aug",
    "september",
    "sep",
    "october",
    "oct",
    "november",
    "nov",
    "december",
    "dec",
    "metre",
    "meter",
    "kilometer",
    "centimeter",
    "millimeter",
    "micrometer",
    "nanometer",
    "inch",
    "foot",
    "feet",
    "yard",
    "mile",
    "kilogram",
    "gram",
    "milligram",
    "microgram",
    "tonne",
    "ton",
    "pound",
    "ounce",
    "stone",
    "carat",
    "grain",
    "cent",
    "dollar",
    "euro",
    "pound",
    "yen",
    "celsius",
    "fahrenheit",
    "kelvin",
    "square",
    "acre",
    "hectare",
    "cubic",
    "liter",
    "milliliter",
    "gallon",
    "quart",
    "pint",
    "degree",
    "radian",
    "rad",
    "percent",
    "south",
    "north",
    "east",
    "west",
    "vote",
    "passenger",
    "fathom",
    "intermediate",
    "people",
    "button",
    "line",
    "stitch",
    "edge",
    "time",
    "vols.",
]

pre_units = [
    "$",
    "€",
    "£",
    "¥",
]

post_units = [
    "°",
    "%",
    "s",
    "ns",
    "ms",
    "min",
    "h",
    "d",
    "wk",
    "mo",
    "yr",
    "dec",
    "cent",
    "m",
    "km",
    "cm",
    "mm",
    "nm",
    "in",
    "ft",
    "yd",
    "mi",
    "ly",
    "kg",
    "g",
    "mg",
    "t",
    "tn",
    "lb",
    "oz",
    "st",
    "ct",
    "gr",
    "ha",
    "ac",
    "l",
    "ml",
    "gal",
    "qt",
    "pt",
    "cc",
    "°c",
    "°f",
    "k",
    "hz",
]

del_start_phrases = [
    "footnote",
    "note",
    "illustration",
    "sidenote",
    "page",
]

del_mid_phrases = [
    "p.",
    "page",
    "Page",
    "volumes",
    "vol.",
    "Vol.",
    "edition",
    "ed.",
    "Edition",
    "Ed.",
]


class TextNormalizer:
    def __init__(self):
        self.en_tn_model = EnNormalizer(cache_dir="/tmp/tn", overwrite_cache=False)
        self.table = str.maketrans("’‘，。；？！（）：-《》、“”【】_", "'',.;?!(): <>/\"\"[] ")

    def __call__(self, cut):
        text = cut["supervisions"][0]["custom"]["texts"][0]

        text = text.translate(self.table)

        text = re.sub(r"\(\d+\)|\{\d+\}|\[\d+\]|<\d+>", " ", text)

        text = re.sub(r"\[FN#\d+\]", " ", text)

        del_start_pattern = rf"(?i)[\{{\[<\(]\s*({'|'.join(del_start_phrases)})\b.*?[\}}>\]\)]|[\{{\[<\(]\s*({'|'.join(del_start_phrases)})\b.*?$"
        text = re.sub(del_start_pattern, " ", text)

        pattern = r"\([^\)]*?\d+[^\)]*?\)|\{[^\}]*?\d+[^\}]*?\}|\[[^\]]*?\d+[^\]]*?\]|<[^>]*?\d+[^>]*?>"
        del_mid_pattern = (
            r"(?:(?:^|\s)(?:" + "|".join(map(re.escape, del_mid_phrases)) + r")\b)"
        )
        unit_pattern = (
            r"(?i)\b("
            + "|".join([re.escape(unit) + r"(?:s|es)?" for unit in units])
            + r")\b"
        )
        pre_units_pattern = r"(?i)(" + "|".join(map(re.escape, pre_units)) + r")\d+"
        post_units_pattern = r"(?i)\d+(" + "|".join(map(re.escape, post_units)) + r")"

        if (match := re.search(pattern, text)) is not None:
            content = match.group(0)
            if re.search(del_mid_pattern, content) or not (
                re.search(unit_pattern, content)
                or re.search(pre_units_pattern, content)
                or re.search(post_units_pattern, content)
            ):
                text = text.replace(content, " ")

        text = re.sub(r"\b\d+:\d{3,}\b", "", text)
        text = re.sub(r"\b\d+:\b\d+:\d{3,}\b", "", text)

        text = re.sub(r"\\\"", "", text)
        text = re.sub(r"\\\'", "", text)
        text = re.sub(r"\\", "", text)

        text = re.sub(r"\.{3,}", "…", text)
        text = re.sub(r"[^\w\s.,!?;:…\'']", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        if len(text) == 0:
            return None

        text = self.en_tn_model.normalize(text)

        cut["supervisions"][0]["text"] = text
        del cut["supervisions"][0]["custom"]
        del cut["custom"]

        return cut


def main():
    assert len(sys.argv) == 3, "Usage: ./local/prepare_manifest.py INPUT OUTPUT_DIR"
    fname = Path(sys.argv[1]).name
    oname = Path(sys.argv[2]) / fname

    tn = TextNormalizer()

    cuts = set()
    if oname.exists():
        with gzip.open(oname, "r") as fin:
            for line in tqdm(fin, desc="Loading processed"):
                cuts.add(json.loads(line)["id"])

    with ProcessPoolExecutor() as ex:
        with gzip.open(sys.argv[1], "r") as fin:
            futures = []
            for line in tqdm(fin, desc="Distributing"):
                parsed_line = json.loads(line)
                if parsed_line["id"] not in cuts:
                    futures.append(ex.submit(tn, parsed_line))

        with gzip.open(oname, "a") as fout:
            for future in tqdm(futures, desc="Processing"):
                try:
                    result = future.result()
                    if result is not None:
                        fout.write((json.dumps(result) + "\n").encode())
                except Exception as e:
                    print(f"Caught exception:\n{e}\n")


if __name__ == "__main__":
    main()
