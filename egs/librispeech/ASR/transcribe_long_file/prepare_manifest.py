#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This script generates manifests for given audio directories.
"""

import argparse
import logging
from pathlib import Path

from lhotse import Recording, RecordingSet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to directory to save the manifests.",
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/audio"),
        help="Path to directory that save audio files.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))

    audio_dir = args.audio_dir

    manifest_dir = args.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    audio_suffix = ".flac"
    json_suffix = ".jsonl.gz"
    subsets = ["librispeech_cuts_test-clean"]

    for subset in subsets:
        logging.info(f"Processing {subset}")

        manifest_out = manifest_dir / (subset + json_suffix)
        if manifest_out.is_file():
            logging.info(f"{manifest_out} already exists - skipping.")
            continue

        recordings = []
        subset_dir = audio_dir / subset
        for audio_path in subset_dir.glob("*" + audio_suffix):
            rec = Recording.from_file(audio_path)
            recordings.append(rec)

        recording_set = RecordingSet.from_recordings(recordings)
        recording_set.to_file(manifest_out)

        logging.info(f"Recordings saved to {manifest_out}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
