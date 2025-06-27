#!/usr/bin/python
# Johns Hopkins University  (authors: Amir Hussein)

"""
Sample data given duration in seconds.
"""

import argparse
import logging
import os
from pathlib import Path

from lhotse import CutSet, RecordingSet, SupervisionSet


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--sup",
        type=str,
        default="",
        help="Supervisions file",
    )

    parser.add_argument(
        "--rec",
        type=str,
        default="",
        help="Recordings file",
    )
    parser.add_argument(
        "--cut",
        type=str,
        default="",
        help="Cutset file",
    )
    parser.add_argument(
        "--outcut",
        type=str,
        default="",
        help="name of the cutset to be saved",
    )
    parser.add_argument(
        "--dur",
        type=float,
        default=10.0,
        help="duration of the cut in seconds",
    )

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    if args.cut != "":
        logging.info(f"Loading {args.cut}")
        cuts = CutSet.from_file(args.cut)
        outdir = Path(os.path.dirname(args.cut))

    else:
        outdir = Path(os.path.dirname(args.sup))
        logging.info(f"Loading supervisions")
        recordings = RecordingSet.from_file(args.rec)
        supervisions = SupervisionSet.from_file(args.sup)
        logging.info("Fixing manifests")
        cuts = CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )
        cuts = cuts.trim_to_supervisions(
            keep_overlapping=False, keep_all_channels=False
        )

    shuffled = cuts.shuffle()
    total_dur = 0
    cuts_list = []
    for c in shuffled:
        if total_dur < args.dur and "_sp" not in c.id:
            total_dur += c.duration
            cuts_list.append(c.id)
        else:
            break
    cuts = cuts.filter(lambda c: c.id in cuts_list)
    cuts.describe()

    logging.info(f"Saving {args.outcut}")
    cuts.to_file(outdir / args.outcut)


if __name__ == "__main__":
    main()
