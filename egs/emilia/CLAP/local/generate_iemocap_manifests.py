import argparse
import glob
import logging
import os

import torch
from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

EMOTIONS = ["ang", "hap", "neu", "exc", "sad"]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the iemocap dataset",
        default="./download/IEMOCAP",
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

    for session_id in [1, 2, 3, 4, 5]:
        wav_folder = f"{dataset_dir}/Session{session_id}/dialog/wav"
        label_folder = f"{dataset_dir}/Session{session_id}/dialog/EmoEvaluation"

        label_files = sorted(glob.glob(f"{label_folder}/Ses*.txt"))

        dataset = {}

        for label in label_files:
            with open(label, "r") as f:
                data = f.readlines()

            for line in data:
                # skip lines
                if line[0] != "[":
                    continue
                items = line.strip().split("\t")
                timestamp = items[0].replace("[", "").replace("]", "").split()
                timestamp = [float(timestamp[0]), float(timestamp[2])]
                clip_name = items[1]
                audio_name = clip_name.rsplit("_", 1)[0]
                emotion = items[2]
                audio_name = wav_folder + "/" + f"{audio_name}.wav"

                assert os.path.isfile(audio_name)
                assert clip_name not in dataset

                dataset[clip_name] = [audio_name, timestamp, emotion]

        logging.info(f"A total of {len(dataset)} clips!")

        cuts = []
        for i, (cut_id, info) in enumerate(dataset.items()):
            audio_file, timestamp, emotion = info
            recording = Recording.from_file(audio_file, cut_id)
            if emotion not in EMOTIONS:
                continue
            if emotion == "exc":
                emotion = "hap"
            assert recording.sampling_rate == 16000
            cut = MonoCut(
                id=cut_id,
                start=timestamp[0],
                duration=timestamp[1] - timestamp[0],
                channel=0,
                recording=recording,
            )
            supervision = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0.0,
                channel=0,
                duration=cut.duration,
                text="",
            )
            supervision.emotion = emotion

            cut.supervisions = [supervision]
            cuts.append(cut)

            if i % 100 == 0 and i:
                logging.info(f"Processed {i} cuts until now.")

        logging.info(f"After filtering, a total of {len(cuts)} valid samples.")
        cuts = CutSet.from_cuts(cuts)

        manifest_output_dir = (
            manifest_dir + "/" + f"iemocap_cuts_session{session_id}.jsonl.gz"
        )

        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cuts.to_jsonl(manifest_output_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
