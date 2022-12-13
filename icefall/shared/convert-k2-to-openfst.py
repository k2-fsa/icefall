#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This script takes as input an FST in k2 format and convert it
to an FST in OpenFST format.

The generated FST is saved into a binary file and its type is
StdVectorFst.

Usage examples:
(1) Convert an acceptor

  ./convert-k2-to-openfst.py in.pt binary.fst

(2) Convert a transducer

  ./convert-k2-to-openfst.py --olabels aux_labels in.pt binary.fst
"""

import argparse
import logging
from pathlib import Path

import k2
import kaldifst.utils
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--olabels",
        type=str,
        default=None,
        help="""If not empty, the input FST is assumed to be a transducer
        and we use its attribute specified by "olabels" as the output labels.
        """,
    )
    parser.add_argument(
        "input_filename",
        type=str,
        help="Path to the input FST in k2 format",
    )

    parser.add_argument(
        "output_filename",
        type=str,
        help="Path to the output FST in OpenFst format",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.info(f"{vars(args)}")

    input_filename = args.input_filename
    output_filename = args.output_filename
    olabels = args.olabels

    if Path(output_filename).is_file():
        logging.info(f"{output_filename} already exists - skipping")
        return

    assert Path(input_filename).is_file(), f"{input_filename} does not exist"
    logging.info(f"Loading {input_filename}")
    k2_fst = k2.Fsa.from_dict(torch.load(input_filename))
    if olabels:
        assert hasattr(k2_fst, olabels), f"No such attribute: {olabels}"

    p = Path(output_filename).parent
    if not p.is_dir():
        logging.info(f"Creating {p}")
        p.mkdir(parents=True)

    logging.info("Converting (May take some time if the input FST is large)")
    fst = kaldifst.utils.k2_to_openfst(k2_fst, olabels=olabels)
    logging.info(f"Saving to {output_filename}")
    fst.write(output_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
