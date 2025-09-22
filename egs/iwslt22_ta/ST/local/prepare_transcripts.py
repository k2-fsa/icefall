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
        "--src-langdir",
        type=str,
        default="",
        help="name of the source lang-dir",
    )
    parser.add_argument(
        "--tgt-langdir",
        type=str,
        default=None,
        help="name of the target lang-dir",
    )
    return parser
     

def main():

    parser = get_parser()
    args = parser.parse_args()

    logging.info("Reading the cuts")
    cuts = CutSet.from_file(args.cut)
    if args.tgt_langdir != None:
        logging.info("Target dir is not None")
        langdirs = [Path(args.src_langdir), Path(args.tgt_langdir)]
    else:
        langdirs = [Path(args.src_langdir)]
    
    for langdir in langdirs:
        if not os.path.exists(langdir):
            os.makedirs(langdir)

    with open(langdirs[0] / "transcript_words.txt", 'w') as src, open(langdirs[1] / "transcript_words.txt", 'w') as tgt:
        for c in cuts:
            src_txt = c.supervisions[0].text
            tgt_txt = c.supervisions[0].custom['translated_text']['eng']
            src.write(src_txt + '\n')
            tgt.write(tgt_txt + '\n')

if __name__ == "__main__":
    main()