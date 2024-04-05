#!/usr/bin/python

from lhotse import RecordingSet, SupervisionSet, CutSet
import argparse
import logging
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
import pdb



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
        "--savecut",
        type=str,
        default="",
        help="name of the cutset to be saved",
    )

    return parser


def valid_asr(cut):
    tol = 2e-3
    i=0
    total_dur = 0
    for c in cut:
        if c.supervisions != []:
            if c.supervisions[0].end > c.duration + tol:

                logging.info(f"Supervision beyond the cut. Cut number: {i}")
                total_dur += c.duration
                logging.info(f"id: {c.id}, sup_end: {c.supervisions[0].end},  dur: {c.duration}, source {c.recording.sources[0].source}")
            elif c.supervisions[0].start < -tol:
                logging.info(f"Supervision starts before the cut. Cut number: {i}")
                logging.info(f"id: {c.id}, sup_start: {c.supervisions[0].start},  dur: {c.duration}, source {c.recording.sources[0].source}")
            else:
                continue
        else:
            logging.info("Empty supervision")
            logging.info(f"id: {c.id}")
        i += 1
    logging.info(f"filtered duration: {total_dur}")
     

def main():

    parser = get_parser()
    args = parser.parse_args()
    if args.cut != "":
        cuts = CutSet.from_file(args.cut)
    else:
        recordings = RecordingSet.from_file(args.rec)
        supervisions = SupervisionSet.from_file(args.sup)
       # breakpoint()
        logging.info("Example from supervisions:")
        logging.info(supervisions[0])
        logging.info("Example from recordings")
        logging.info("Fixing manifests")
        recordings, supervisions = fix_manifests(recordings, supervisions)
        logging.info("Validating manifests")
        validate_recordings_and_supervisions(recordings, supervisions)
    
        cuts = CutSet.from_manifests(recordings= recordings, supervisions=supervisions,)
    cuts = cuts.trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)
    cuts.describe()
    logging.info("Example from cut:")
    logging.info(cuts[100])
    logging.info("Validating manifests for ASR")
    valid_asr(cuts)
    if args.savecut != "":
        cuts.to_file(args.savecut)

if __name__ == "__main__":
    main()