# Copyright 2023 Johns Hopkins University  (Amir Hussein)

#!/usr/bin/python
"""
This script prepares transcript_words.txt from cutset
"""

from lhotse import CutSet
import argparse
import logging
import pdb
from pathlib import Path
import os


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cut",
        type=str,
        default="",
        help="Cutset file",
    )
    parser.add_argument(
        "--langdir",
        type=str,
        default="",
        help="name of the lang-dir",
    )
    return parser
     

def main():

    parser = get_parser()
    args = parser.parse_args()

    logging.info("Reading the cuts")
    cuts = CutSet.from_file(args.cut)
    langdir = Path(args.langdir)
    
    if not os.path.exists(langdir):
        os.makedirs(langdir)
    
    with open(langdir / "st_words.txt", 'w') as txt:
        for c in cuts:
            text = c.supervisions[0].custom['translated_text']['en']
            txt.write(text + '\n')

if __name__ == "__main__":
    main()