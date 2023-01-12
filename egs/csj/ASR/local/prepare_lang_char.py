#!/usr/bin/env python3
# Copyright    2022  The University of Electro-Communications  (Author: Teo Wen Shen)  # noqa
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
import logging
from pathlib import Path

from lhotse import CutSet
from lhotse.recipes.csj import CSJSDBParser

ARGPARSE_DESCRIPTION = """
This script gathers all training transcripts, parses them in disfluent mode, and produces a token_list that would be the output set of the ASR system.

It splits transcripts by whitespace into lists, then, for each word in the
list, if the word does not appear in the list of user-defined multicharacter
strings, it further splits that word into individual characters to be counted
into the output token set.

It outputs 3 files into the lang directory:
- userdef_string: a list of user defined strings that should not be split
 further into individual characters. By default, it contains "<unk>", "<blk>",
 "<sos/eos>"
- tokens_len: the total number of tokens in the output set.
- tokens.txt: a list of tokens in the output set. The length matches tokens_len.

"""


def get_args():
    parser = argparse.ArgumentParser(
        description=ARGPARSE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "train_cut", metavar="train-cut", type=Path, help="Path to the train cut"
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=None,
        help=("Name of lang dir. " "If not set, this will default to lang_char"),
    )

    parser.add_argument(
        "--userdef-string",
        type=Path,
        default=None,
        help="Multicharacter strings that do not need to be split",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        format=("%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"),
        level=logging.INFO,
    )

    if not args.lang_dir:
        p = "lang_char"
        args.lang_dir = Path(p)

    if args.userdef_string:
        args.userdef_string = set(args.userdef_string.read_text().split())
    else:
        args.userdef_string = set()

    sysdef_string = set(["<blk>", "<unk>", "<sos/eos>"])
    args.userdef_string.update(sysdef_string)

    # Using disfluent parsing as fluent is a subset of disfluent
    parser = CSJSDBParser()

    token_set = set()
    logging.info(f"Creating vocabulary from {args.train_cut}.")
    train_cut: CutSet = CutSet.from_file(args.train_cut)
    for cut in train_cut:
        if "_sp" in cut.id:
            continue

        text: str = cut.supervisions[0].custom["raw"]
        for w in parser.parse(text, sep=" ").split(" "):
            if w in args.userdef_string:
                token_set.add(w)
            else:
                token_set.update(w)

    token_set = ["<blk>"] + sorted(token_set - sysdef_string) + ["<unk>", "<sos/eos>"]
    args.lang_dir.mkdir(parents=True, exist_ok=True)
    (args.lang_dir / "tokens.txt").write_text(
        "\n".join(f"{t}\t{i}" for i, t in enumerate(token_set))
    )
    (args.lang_dir / "tokens_len").write_text(f"{len(token_set)}")

    (args.lang_dir / "userdef_string").write_text("\n".join(args.userdef_string))
    logging.info("Done.")


if __name__ == "__main__":
    main()
