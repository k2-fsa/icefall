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

ARGPARSE_DESCRIPTION = """
This script gathers all training transcripts of the specified {trans_mode} type
and produces a token_list that would be output set of the ASR system.

It splits transcripts by whitespace into lists, then, for each word in the
list, if the word does not appear in the list of user-defined multicharacter
strings, it further splits that word into individual characters to be counted
into the output token set.

It outputs 4 files into the lang directory:
- trans_mode: the name of transcript mode. If trans_mode was not specified,
   this will be an empty file.
- userdef_string: a list of user defined strings that should not be split
 further into individual characters. By default, it contains "<unk>", "<blk>",
 "<sos/eos>"
- words_len: the total number of tokens in the output set.
- words.txt: a list of tokens in the output set. The length matches words_len.

"""


def get_args():
    parser = argparse.ArgumentParser(
        description=ARGPARSE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-cut", type=Path, required=True, help="Path to the train cut"
    )

    parser.add_argument(
        "--trans-mode",
        type=str,
        default=None,
        help=(
            "Name of the transcript mode to use. "
            "If lang-dir is not set, this will also name the lang-dir"
        ),
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=None,
        help=(
            "Name of lang dir. "
            "If not set, this will default to lang_char_{trans-mode}"
        ),
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
        if args.trans_mode:
            p += f"_{args.trans_mode}"
        args.lang_dir = Path(p)

    if args.userdef_string:
        args.userdef_string = set(args.userdef_string.read_text().split())
    else:
        args.userdef_string = set()

    sysdef_string = ["<blk>", "<unk>", "<sos/eos>"]
    args.userdef_string.update(sysdef_string)

    train_set: CutSet = CutSet.from_file(args.train_cut)

    words = set()
    logging.info(
        f"Creating vocabulary from {args.train_cut.name} at {args.trans_mode} mode."
    )
    for cut in train_set:
        try:
            text: str = (
                cut.supervisions[0].custom[args.trans_mode]
                if args.trans_mode
                else cut.supervisions[0].text
            )
        except KeyError:
            raise KeyError(
                f"Could not find {args.trans_mode} in {cut.supervisions[0].custom}"
            )
        for t in text.split():
            if t in args.userdef_string:
                words.add(t)
            else:
                words.update(c for c in list(t))

    words -= set(sysdef_string)
    words = sorted(words)
    words = ["<blk>"] + words + ["<unk>", "<sos/eos>"]

    args.lang_dir.mkdir(parents=True, exist_ok=True)
    (args.lang_dir / "words.txt").write_text(
        "\n".join(f"{word}\t{i}" for i, word in enumerate(words))
    )

    (args.lang_dir / "words_len").write_text(f"{len(words)}")

    (args.lang_dir / "userdef_string").write_text("\n".join(args.userdef_string))

    (args.lang_dir / "trans_mode").write_text(args.trans_mode)
    logging.info("Done.")


if __name__ == "__main__":
    main()
