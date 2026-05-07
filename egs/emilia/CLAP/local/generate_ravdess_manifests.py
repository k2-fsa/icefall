import argparse
import glob
import json
import logging
import os

import torch
from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "sad",
    "surprised",
    "neutral",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the ravdess dataset",
        default="./download/ravdess",
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    os.makedirs(manifest_dir, exist_ok=True)

    for fold_id in [0, 1, 2, 3]:
        dataset = {}

        label_paths = sorted(glob.glob(f"{dataset_dir}/fold_{fold_id}/*.json"))
        for label_path in label_paths:
            with open(label_path, "r") as f:
                item = json.load(f)
            emotion = item["emotion"]

            audio_path = label_path.replace(".json", ".wav")
            assert os.path.isfile(audio_path)
            audio_name = audio_path.split("/", 2)[-1].replace(".wav", "")

            speaker_id = int(audio_name.rsplit("-", 1)[-1])
            if speaker_id % 2 == 0:
                gender = "female"
            else:
                gender = "male"

            dataset[audio_name] = [audio_path, gender, emotion]

        logging.info(f"A total of {len(dataset)} clips!")

        cuts = []
        for i, (cut_id, info) in enumerate(dataset.items()):
            audio_path, gender, emotion = info
            recording = Recording.from_file(audio_path, cut_id)
            cut = MonoCut(
                id=cut_id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )
            supervision = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0,
                channel=0,
                duration=cut.duration,
                text="",
                gender=gender,
            )
            supervision.emotion = emotion

            cut.supervisions = [supervision]
            cut = cut.resample(16000)

            cuts.append(cut)

            if i % 100 == 0 and i:
                logging.info(f"Processed {i} cuts until now.")

        cuts = CutSet.from_cuts(cuts)

        manifest_output_dir = (
            manifest_dir + "/" + f"ravdess_cuts_fold{fold_id}.jsonl.gz"
        )

        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cuts.to_jsonl(manifest_output_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
