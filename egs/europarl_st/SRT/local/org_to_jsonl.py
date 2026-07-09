# Copyright 2026 Nanjie Li (linanjie0820@gmail.com)
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
Preprocess the Europarl-ST dataset into per-language-pair JSONL files.

Reference: https://www.mllp.upv.es/europarl-st/
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.audio_utils import audio_to_flac
from utils.dataset_parameters import AUDIO_SAVE_SAMPLE_RATE


def read_lst_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    return lines


def parse_timestamp(timestamp):
    if ":" in timestamp:
        parts = [float(part) for part in timestamp.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
        return hours * 3600 + minutes * 60 + seconds
    return float(timestamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw Europarl-ST data to per-language-pair JSONL files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "v1.1"),
        help="Path to the Europarl-ST v1.1 raw data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "audio"
        ),
        help="Directory to store converted FLAC audio segments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = os.path.realpath(args.data_dir)
    output_dir = os.path.realpath(args.output_dir)

    # Remap original splits: train/dev/test -> train/valid/test.
    split_output_name_dict = {
        "train": "train",
        "dev": "valid",
        "test": "test",
        # 'train-noisy': 'train-noisy'  # Skipped: this split contains many errors.
    }

    languages = ["es", "de", "en", "fr", "nl", "pl", "pt", "ro", "it"]

    texts_output_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "texts")
    )
    os.makedirs(texts_output_dir, exist_ok=True)

    pair_jsonl_paths = {}
    dataset_prefix = "europarl"

    for src_lang in languages:
        for dest_lang in languages:
            if src_lang == dest_lang:
                continue
            pair_dir = os.path.join(texts_output_dir, f"{src_lang}_{dest_lang}")
            os.makedirs(pair_dir, exist_ok=True)
            for split_alias in split_output_name_dict.values():
                jsonl_path = os.path.join(
                    pair_dir,
                    f"{dataset_prefix}.{src_lang}_{dest_lang}.{split_alias}.jsonl",
                )
                with open(jsonl_path, "w", encoding="utf-8"):
                    pass
                pair_jsonl_paths[(src_lang, dest_lang, split_alias)] = jsonl_path

    dict_skeleton = {
        "es": None,
        "de": None,
        "en": None,
        "fr": None,
        "nl": None,
        "pl": None,
        "pt": None,
        "ro": None,
        "it": None,
    }

    file_ids = {"train": 1, "dev": 1, "test": 1}

    for source_lang in languages:

        print(f"Processing {source_lang} dataset...\n")

        language_folder = os.path.join(data_dir, source_lang)

        destination_languages = languages.copy()
        destination_languages.remove(source_lang)

        for split, split_name in split_output_name_dict.items():

            os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
            segments_dict = {}

            for dest_lang in destination_languages:

                segments_lst_file = os.path.join(
                    language_folder, dest_lang, split, "segments.lst"
                )
                segments_source_lang_file = os.path.join(
                    language_folder, dest_lang, split, f"segments.{source_lang}"
                )
                segments_dest_lang_file = os.path.join(
                    language_folder, dest_lang, split, f"segments.{dest_lang}"
                )

                segments_timestamps = read_lst_file(segments_lst_file)
                segments_source_lang_transcriptions = read_lst_file(
                    segments_source_lang_file
                )
                segments_dest_lang_transcriptions = read_lst_file(
                    segments_dest_lang_file
                )

                segments_source_lang_transcriptions_dict = dict(
                    zip(segments_timestamps, segments_source_lang_transcriptions)
                )
                segments_dest_lang_transcriptions_dict = dict(
                    zip(segments_timestamps, segments_dest_lang_transcriptions)
                )

                for segment in segments_timestamps:

                    segments_dict.setdefault(segment, dict_skeleton.copy())
                    segments_dict[segment][
                        source_lang
                    ] = segments_source_lang_transcriptions_dict[segment]
                    segments_dict[segment][
                        dest_lang
                    ] = segments_dest_lang_transcriptions_dict[segment]

            for segment in segments_dict:
                audio, segment_start, segment_end = segment.split()

                audio_path = os.path.join(language_folder, "audios", f"{audio}.m4a")
                audio_output_filename = f"{source_lang}_{file_ids[split]}.flac"
                audio_output_path = os.path.join(
                    output_dir, split_name, audio_output_filename
                )

                audio_to_flac(
                    audio_path,
                    audio_output_path,
                    sample_rate=AUDIO_SAVE_SAMPLE_RATE,
                    segment_start=segment_start,
                    segment_end=segment_end,
                )

                source_transcription = segments_dict[segment][source_lang]
                if source_transcription in (None, "None"):
                    file_ids[split] += 1
                    continue

                translations = {
                    lang: text
                    for lang, text in segments_dict[segment].items()
                    if lang != source_lang and text not in (None, "None")
                }

                if not translations:
                    file_ids[split] += 1
                    continue

                try:
                    duration_seconds = parse_timestamp(segment_end) - parse_timestamp(
                        segment_start
                    )
                except ValueError:
                    file_ids[split] += 1
                    continue

                if duration_seconds <= 0:
                    file_ids[split] += 1
                    continue

                rounded_duration = round(duration_seconds, 3)

                for dest_lang, translation in translations.items():
                    jsonl_path = pair_jsonl_paths[(source_lang, dest_lang, split_name)]
                    jsonl_entry = {
                        "source": audio_output_path,
                        "duration": rounded_duration,
                        "text": source_transcription,
                        "st_text": translation,
                    }
                    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
                        jsonl_file.write(
                            json.dumps(jsonl_entry, ensure_ascii=False) + "\n"
                        )

                file_ids[split] += 1
