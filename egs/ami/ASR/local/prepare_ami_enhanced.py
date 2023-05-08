#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for AMI GSS-enhanced dataset.

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lhotse import Recording, RecordingSet, SupervisionSet
from lhotse.qa import fix_manifests
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.utils import fastcopy
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="AMI enhanced dataset preparation.")
    parser.add_argument(
        "manifests_dir",
        type=Path,
        help="Path to directory containing AMI manifests.",
    )
    parser.add_argument(
        "enhanced_dir",
        type=Path,
        help="Path to enhanced data directory.",
    )
    parser.add_argument(
        "--num-jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--min-segment-duration",
        "-d",
        type=float,
        default=0.0,
        help="Minimum duration of a segment in seconds.",
    )
    return parser.parse_args()


def find_recording_and_create_new_supervision(enhanced_dir, supervision):
    """
    Given a supervision (corresponding to original AMI recording), this function finds the
    enhanced recording correspoding to the supervision, and returns this recording and
    a new supervision whose start and end times are adjusted to match the enhanced recording.
    """
    file_name = Path(
        f"{supervision.recording_id}-{supervision.speaker}-{int(100*supervision.start):06d}_{int(100*supervision.end):06d}.flac"
    )
    save_path = enhanced_dir / f"{supervision.recording_id}" / file_name
    if save_path.exists():
        recording = Recording.from_file(save_path)
        if recording.duration == 0:
            logging.warning(f"Skipping {save_path} which has duration 0 seconds.")
            return None

        # Old supervision is wrt to the original recording, we create new supervision
        # wrt to the enhanced segment
        new_supervision = fastcopy(
            supervision,
            recording_id=recording.id,
            start=0,
            duration=recording.duration,
        )
        return recording, new_supervision
    else:
        logging.warning(f"{save_path} does not exist.")
        return None


def main(args):
    # Get arguments
    manifests_dir = args.manifests_dir
    enhanced_dir = args.enhanced_dir

    # Load manifests from cache if they exist (saves time)
    manifests = read_manifests_if_cached(
        dataset_parts=["train", "dev", "test"],
        output_dir=manifests_dir,
        prefix="ami-sdm",
        suffix="jsonl.gz",
    )
    if not manifests:
        raise ValueError("AMI SDM manifests not found in {}".format(manifests_dir))

    with ThreadPoolExecutor(args.num_jobs) as ex:
        for part in ["train", "dev", "test"]:
            logging.info(f"Processing {part}...")
            supervisions_orig = manifests[part]["supervisions"].filter(
                lambda s: s.duration >= args.min_segment_duration
            )
            # Remove TS3009d supervisions since they are not present in the enhanced data
            supervisions_orig = supervisions_orig.filter(
                lambda s: s.recording_id != "TS3009d"
            )
            futures = []

            for supervision in tqdm(
                supervisions_orig,
                desc="Distributing tasks",
            ):
                futures.append(
                    ex.submit(
                        find_recording_and_create_new_supervision,
                        enhanced_dir,
                        supervision,
                    )
                )

            recordings = []
            supervisions = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc="Processing tasks",
            ):
                result = future.result()
                if result is not None:
                    recording, new_supervision = result
                    recordings.append(recording)
                    supervisions.append(new_supervision)

            # Remove duplicates from the recordings
            recordings_nodup = {}
            for recording in recordings:
                if recording.id not in recordings_nodup:
                    recordings_nodup[recording.id] = recording
                else:
                    logging.warning("Recording {} is duplicated.".format(recording.id))
            recordings = RecordingSet.from_recordings(recordings_nodup.values())
            supervisions = SupervisionSet.from_segments(supervisions)

            recordings, supervisions = fix_manifests(
                recordings=recordings, supervisions=supervisions
            )

            logging.info(f"Writing {part} enhanced manifests")
            recordings.to_file(manifests_dir / f"ami-gss_recordings_{part}.jsonl.gz")
            supervisions.to_file(
                manifests_dir / f"ami-gss_supervisions_{part}.jsonl.gz"
            )


if __name__ == "__main__":
    args = get_args()
    main(args)
