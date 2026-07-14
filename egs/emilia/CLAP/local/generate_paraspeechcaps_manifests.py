import argparse
import json
import logging
import os
import re
import tarfile

from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment
from normalize_paraspeechcaps_short_captions import normalize


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
    )

    return parser.parse_args()


def process_psc_base(args, subset, source):
    manifests_file = f"{args.output_dir}/paraspeechcaps_{subset}-{source}.jsonl"
    output_path = (
        args.output_dir + "/" + f"paraspeechcaps_cuts_{subset}-{source}.jsonl.gz"
    )

    if os.path.exists(output_path):
        print(f"{output_path} exists, skip")
        return

    cuts = []
    num_cuts = 0

    logging.info(f"Loading manifest: {manifests_file}")
    with open(manifests_file) as reader:
        for line in reader:
            item = json.loads(line)

            audio_path = item["audio_path"]

            speaker = item["speaker"].strip()
            gender = item["gender"].strip()
            accent = item["accent"].strip()
            pitch = item["pitch"].strip()
            speaking_rate = item["speaking_rate"].strip()
            intrinsic_tags = [i.strip() for i in item["intrinsic_tags"]]
            situational_tags = (
                [i.strip() for i in item["situational_tags"]]
                if item["situational_tags"] is not None
                else []
            )

            transcription = item["text"].strip()
            short_captions = [
                normalize(re.sub(r"[\t\n\r]", " ", i).strip(), accent)
                for i in item["caption"]
            ]

            cut_id = (
                subset
                + "-"
                + source
                + "-"
                + audio_path.replace("download/", "")
                .replace("/", "-")
                .replace(".wav", "")
            )

            if not os.path.exists(audio_path):
                logging.warning(f"No such file: {audio_path}")
                continue

            recording = Recording.from_file(audio_path, cut_id)
            cut = MonoCut(
                id=cut_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )

            supervision = SupervisionSegment(
                id=recording.id,
                recording_id=recording.id,
                start=0.0,
                channel=0,
                duration=recording.duration,
                text=transcription,
                speaker=speaker,
            )
            supervision.short_captions = short_captions
            supervision.long_captions = []

            supervision.gender = gender
            supervision.accent = accent
            supervision.pitch = pitch
            supervision.speaking_rate = speaking_rate
            supervision.intrinsic_tags = intrinsic_tags
            supervision.situational_tags = situational_tags

            cut.supervisions = [supervision]
            cut = cut.resample(16000)
            cuts.append(cut)

            num_cuts += 1
            if num_cuts % 100 == 0 and num_cuts:
                logging.info(f"Processed {num_cuts} cuts until now.")

    cut_set = CutSet.from_cuts(cuts)

    logging.info(f"Saving to {output_path}")
    cut_set.to_file(output_path)


def process_psc_scaled(args, subset, source):
    manifests_file = f"{args.output_dir}/paraspeechcaps_{subset}-{source}.jsonl"
    output_path = (
        args.output_dir + "/" + f"paraspeechcaps_cuts_{subset}-{source}.jsonl.gz"
    )

    if os.path.exists(output_path):
        print(f"{output_path} exists, skip")
        return

    items = []
    logging.info(f"Loading manifest: {manifests_file}")
    with open(manifests_file) as reader:
        for line in reader:
            items.append(json.loads(line))

    def extract_key(audio_path: str) -> str:
        return "".join(re.search(r"EN_B(\d+)_S\d+(\d)", audio_path).groups())

    items_sorted = sorted(items, key=lambda x: int(extract_key(x["audio_path"])))

    audio_output_dir = "./download/Emilia-audio"
    os.makedirs(audio_output_dir, exist_ok=True)

    cuts = []
    num_cuts = 0
    current_tar_key = None
    current_tar_handle = None
    for item in items_sorted:
        audio_path_in_tar = item["audio_path"].rsplit("/", 1)[-1]
        transcription = item["text"].strip()
        assert len(item["caption"]) == 1, item["caption"]
        short_captions = [re.sub(r"[\t\n\r]", " ", item["caption"][0]).strip()]

        tar_key = extract_key(audio_path_in_tar)
        while True:
            if tar_key != current_tar_key:
                if current_tar_handle:
                    current_tar_handle.close()
                tar_path = f"./download/Emilia/EN/EN-B{tar_key}.tar"
                logging.info(f"About to open tar: {tar_path}")
                current_tar_handle = tarfile.open(tar_path, "r")
                current_tar_key = tar_key

            audio_path = os.path.join(audio_output_dir, audio_path_in_tar)
            try:
                with open(audio_path, "wb") as f:
                    f.write(current_tar_handle.extractfile(audio_path_in_tar).read())
                break
            except:
                logging.warning(
                    f"KeyError: filename {audio_path_in_tar} not found in {tar_path}"
                )
                tar_key = f"{tar_key[:-1] + str((int(tar_key[-1]) + 9) % 10)}"
                continue

        cut_id = subset + "-" + source + "-" + audio_path_in_tar.replace(".mp3", "")

        recording = Recording.from_file(audio_path, cut_id)
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )

        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            channel=0,
            duration=recording.duration,
            text=transcription,
        )
        supervision.short_captions = short_captions
        supervision.long_captions = []

        cut.supervisions = [supervision]
        cut = cut.resample(16000)
        cuts.append(cut)

        num_cuts += 1
        if num_cuts % 100 == 0 and num_cuts:
            logging.info(f"Processed {num_cuts} cuts until now.")

    if current_tar_handle:
        current_tar_handle.close()

    cut_set = CutSet.from_cuts(cuts)

    logging.info(f"Saving to {output_path}")
    cut_set.to_file(output_path)


def main():
    args = get_parser()
    os.makedirs(args.output_dir, exist_ok=True)

    split2subsets = {
        "psc-base": ["test", "dev", "holdout", "train_base"],
        # "psc-scaled": ["train_scaled"],
    }

    split2sources = {
        "psc-base": ["voxceleb", "expresso", "ears"],
        "psc-scaled": ["emilia"],
    }

    for split, subsets in split2subsets.items():
        for subset in subsets:
            for source in split2sources[split]:
                if split == "psc-base":
                    process_psc_base(args, subset, source)
                elif split == "psc-scaled":
                    process_psc_scaled(args, subset, source)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
