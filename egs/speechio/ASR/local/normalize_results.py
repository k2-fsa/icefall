#!/usr/bin/env python3
#
# Copyright 2024 Author: Yuekai Zhang
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
This file uses speech io offcial pipline to normalize the decoding results.
https://github.com/SpeechColab/Leaderboard/blob/master/utils/textnorm_zh.py

Usage:
    python normalize_results.py --model-log-dir ./whisper_decoding_log_dir --output-log-dir ./results_norm
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kaldialign
from speechio_norm import TextNorm

from icefall.utils import store_transcripts, write_error_stats


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-log-dir",
        type=str,
        default="./recogs_whisper",
        help="The directory to store the whisper logs: e.g. recogs-SPEECHIO_ASR_ZH00014-beam-search-epoch--1-avg-1.txt",
    )
    parser.add_argument(
        "--output-log-dir",
        type=str,
        default="./results_whisper_norm",
        help="The directory to store the normalized whisper logs",
    )
    return parser


def save_results_with_speechio_text_norm(
    res_dir: Path,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    normalizer = TextNorm()
    # normlize items in results_dict
    for key, results in results_dict.items():
        results_norm = []
        for item in results:
            wav_name, ref, hyp = item
            ref = normalizer(ref)
            hyp = normalizer(hyp)
            results_norm.append((wav_name, ref, hyp))
        results_dict[key] = results_norm

    test_set_wers = dict()

    suffix = "epoch-999-avg-1"

    for key, results in results_dict.items():
        recog_path = res_dir / f"recogs-{test_set_name}-{key}-{suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        print(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = res_dir / f"errs-{test_set_name}-{key}-{suffix}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        print("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = res_dir / f"wer-summary-{test_set_name}-{key}-{suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    print(s)


def extract_hyp_ref_wavname(filename):
    """
    0Phqz8RWYuE_0007-5:	ref=['R', 'Y', 'Y', 'B', '它最大的优势就是进光量或者说是对光线利用率的提升']
    0Phqz8RWYuE_0007-5:	hyp=而YB它最大的优势是近光量或者说是对光线利用率的提升
    """
    hyps, refs, wav_name = [], [], []
    with open(filename, "r") as f:
        for line in f:
            if "ref" in line:
                ref = line.split("ref=")[1].strip()
                if ref[0] == "[":
                    ref = ref[2:-2]
                list_elements = ref.split("', '")
                ref = "".join(list_elements)
                refs.append(ref)
            elif "hyp" in line:
                hyp = line.split("hyp=")[1].strip()
                hyps.append(hyp)
                wav_name.append(line.split(":")[0])
    return hyps, refs, wav_name


def get_filenames(
    whisper_log_dir,
    whisper_suffix="beam-search-epoch-999-avg-1",
):
    results = []
    start_index, end_index = 0, 26
    dataset_parts = []
    for i in range(start_index, end_index + 1):
        idx = f"{i}".zfill(2)
        dataset_parts.append(f"SPEECHIO_ASR_ZH000{idx}")
    for partition in dataset_parts:
        whisper_filename = f"{whisper_log_dir}/recogs-{partition}-{whisper_suffix}.txt"
        results.append(whisper_filename)
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()
    # mkdir output_log_dir
    Path(args.output_log_dir).mkdir(parents=True, exist_ok=True)
    filenames = get_filenames(args.model_log_dir)
    for filename in filenames:
        hyps, refs, wav_name = extract_hyp_ref_wavname(filename)
        partition_name = filename.split("/")[-1].split("-")[1]

        save_results_with_speechio_text_norm(
            Path(args.output_log_dir),
            partition_name,
            {"norm": list(zip(wav_name, refs, hyps))},
        )

        print(f"Processed {partition_name}")


if __name__ == "__main__":
    main()
