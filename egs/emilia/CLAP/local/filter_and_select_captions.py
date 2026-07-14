import argparse
import logging
import os
import re
from collections import Counter
from pathlib import Path

import lhotse
from lhotse import CutSet, load_manifest_lazy

MULTI_SPEAKER_PATTERN = re.compile(
    r"\b(speakers|first speaker|second speaker)\b",
    re.IGNORECASE,
)

NO_PATTERN = re.compile(
    r"\bno\b|\bnot\b|\bneither\b|\bnor\b|\bfree from\b|\bwith no\b|\bwithout\b|\blacking\b|\brather than\b",
    re.IGNORECASE,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cuts_path", type=Path, help="Path to the input cuts list file."
    )
    parser.add_argument("--tasks", type=Path, help="Path to the input task list file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/manifests"),
        help="Path to the output directory",
    )

    return parser.parse_args()


def validate_short_captions(short_captions, cut_id, min_len=32):
    assert len(short_captions) in (
        1,
        2,
    ), f"short_captions length must be 1 or 2, got {len(short_captions)}"

    for idx, caption in enumerate(short_captions):
        if len(caption) < min_len:
            logging.info(
                f"Filtered cut (id={cut_id}): "
                f"short caption[{idx}] too short (len={len(caption)})"
            )
            return False

    return True


def validate_long_captions(long_captions, cut_id):
    for idx, caption in enumerate(long_captions):
        if MULTI_SPEAKER_PATTERN.search(caption):
            logging.info(f"Filtered cut (id={cut_id}): multi speaker detected")
            return False

    return True


def filter_long_captions(long_captions):
    long_captions = [caption for caption in long_captions if "\n" not in caption]
    long_captions = sorted(long_captions, key=lambda x: len(x))
    long_captions = [caption for caption in long_captions if 128 <= len(caption) <= 768]
    long_captions = [
        caption for caption in long_captions if not NO_PATTERN.search(caption)
    ]
    return long_captions


def main():
    args = get_parser()

    num_long_captions = []

    if args.tasks is not None:
        cuts_paths = [Path(line) for line in args.tasks.read_text().splitlines()]
    else:
        cuts_paths = [args.cuts_path]

    for cuts_path in cuts_paths:
        output_path = (
            args.output_dir
            / f"{cuts_path.name.replace(''.join(cuts_path.suffixes), '')}-selected.jsonl.gz"
        )
        if os.path.exists(output_path):
            print(f"{output_path} exists, skip")
            return

        cuts = load_manifest_lazy(cuts_path)
        logging.info(f"Loading manifest: {cuts_path}")

        filtered_cuts = []
        for cut in cuts:
            short_captions = cut.supervisions[0].short_captions
            if not validate_short_captions(short_captions, cut.id):
                continue

            long_captions = cut.supervisions[0].long_captions
            if not validate_long_captions(long_captions, cut.id):
                continue
            long_captions = filter_long_captions(long_captions)
            if not long_captions:
                continue

            cut.supervisions[0].long_captions = long_captions

            filtered_cuts.append(cut)
            num_long_captions.append(len(long_captions))

        filtered_cuts = CutSet.from_cuts(filtered_cuts)
        logging.info(f"Saving to {output_path}")
        filtered_cuts.to_file(output_path)

    long_counter = Counter(num_long_captions)
    print("Number of long captions distribution:")
    for count in sorted(long_counter.keys()):
        print(f"Length={count}, count={long_counter[count]}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
