import argparse
import logging
import os
from pathlib import Path
from typing import Optional

POSITIONS = ("DA01", "DA02", "DA03", "DA04")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="""Corpus directory path to compute overlap ratio. If None, we will use all""",
    )

    return parser.parse_args()


def compute_overlap_ratio(
    corpus_dir: Optional[str] = None,
):
    corpus_dir = Path(corpus_dir)
    sections = os.listdir(corpus_dir / "train")
    infos = []
    for section in sorted(sections):
        intervals = []
        overlap = 0
        duration = 0
        for position in POSITIONS:
            text_path = (
                corpus_dir / "train" / section / (position + ".TextGrid")
            ).resolve()
            if not text_path.is_file():
                continue

            with open(text_path) as f:
                datalines = f.read().splitlines()

            for dataline in datalines:
                if "xmin =" in dataline:
                    start = float(dataline.split("=")[1].strip())
                elif "xmax =" in dataline:
                    end = float(dataline.split("=")[1].strip())
                elif "text" in dataline:
                    text = dataline.split('"')[1].strip()
                    if len(text) > 0:
                        intervals.append((start, end))
                        duration += end - start

        intervals = sorted(intervals, key=lambda x: x[1])
        st = -1
        ed = -1
        for interval in intervals:
            if ed < interval[0]:  # No overlap
                st = interval[0]
                ed = interval[1]
            else:
                overlap += ed - max(st, interval[0])
                st = min(st, interval[0])
                ed = interval[1]

        infos.append((section, overlap, duration))

    total_overlap = 0
    total_duration = 0
    for info in infos:
        total_overlap += info[1]
        total_duration += info[2]
        logging.info(
            f"section: {info[0]}\t overlap: {info[1]}\t duration: {info[2]}\t overlap ratio: {info[1] / info[2]}"
        )

    logging.info(
        f"total duration: {total_duration}\t total overlap: {total_overlap}\t total overlap ratio: {total_overlap / total_duration}"
    )


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO, filename="overlap_info.log")
    args = get_args()
    logging.info(vars(args))
    compute_overlap_ratio(
        corpus_dir=args.corpus_dir,
    )
    logging.info("Done")


if __name__ == "__main__":
    main()
