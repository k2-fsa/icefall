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
This file uses whisper and zipformer decoding results to generate fusion decoding results.
Since whisper model is more likely to make deletion errors and zipformer model is more likely to make substitution and insertion errors,
we trust whisper model when it makes substitution and insertion errors and trust zipformer model when it makes deletion errors.

Usage:
    python whisper_zipformer_fusion.py --whisper-log-dir ./whisper_decoding_log_dir --zipformer-log-dir ./zipformer_decoding_log_dir --output-log-dir ./results_fusion
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kaldialign

from icefall.utils import store_transcripts, write_error_stats


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--whisper-log-dir",
        type=str,
        default="./recogs_whisper",
        help="The directory to store the whisper logs: e.g. recogs-SPEECHIO_ASR_ZH00014-beam-search-epoch--1-avg-1.txt",
    )
    parser.add_argument(
        "--zipformer-log-dir",
        type=str,
        default="./recogs_zipformer",
        help="The directory to store the zipformer logs",
    )
    parser.add_argument(
        "--output-log-dir",
        type=str,
        default="./results_fusion",
        help="The directory to store the fusion logs",
    )
    return parser


def save_results(
    res_dir: Path,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
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
                ref = ref[2:-2]
                list_elements = ref.split("', '")
                ref = "".join(list_elements)
                refs.append(ref)
            elif "hyp" in line:
                hyp = line.split("hyp=")[1].strip()
                hyps.append(hyp)
                wav_name.append(line.split(":")[0])
    return hyps, refs, wav_name


def get_pair_filenames(
    whisper_log_dir,
    zipformer_log_dir,
    whisper_suffix="beam-search-epoch-999-avg-1",
    zipformer_suffix="greedy_search_blank_penalty_2.0-epoch-999-avg-1-context-2-max-sym-per-frame-1-blank-penalty-2.0",
):
    results = []
    start_index, end_index = 0, 26
    dataset_parts = []
    for i in range(start_index, end_index + 1):
        idx = f"{i}".zfill(2)
        dataset_parts.append(f"SPEECHIO_ASR_ZH000{idx}")
    for partition in dataset_parts:
        whisper_filename = f"{whisper_log_dir}/recogs-{partition}-{whisper_suffix}.txt"
        zipformer_filename = (
            f"{zipformer_log_dir}/recogs-{partition}-{zipformer_suffix}.txt"
        )
        results.append((whisper_filename, zipformer_filename))
    return results


def fusion_hyps_trust_substituion_insertion(
    hyps_whisper, hyps_zipformer, refs, ERR="*"
):
    """
    alignment example:
    [('我', '你'), ('在', '*'), ('任', '任'), ('的', '的'), ('时', '时'), ('候', '候'), ('*', '呢')]
    left is whisper, right is zipformer
    for whisper substitution, use left
    for whisper insertion, use left
    for whisper deletion, use right
    """
    hyps_fusion = []
    for hyp_w, hyp_z, ref in zip(hyps_whisper, hyps_zipformer, refs):
        ali = kaldialign.align(hyp_w, hyp_z, ERR)
        hyp_f = ""
        for a in ali:
            if a[0] == ERR:
                hyp_f += a[1]
            else:
                hyp_f += a[0]
        hyps_fusion.append(hyp_f)
    return hyps_fusion


def fusion_hyps_trust_substituion(hyps_whisper, hyps_zipformer, refs, ERR="*"):
    """
    alignment example:
    [('我', '你'), ('在', '*'), ('任', '任'), ('的', '的'), ('时', '时'), ('候', '候'), ('*', '呢')]
    left is whisper, right is zipformer
    for whisper substitution, use left
    for whisper insertion, use right
    for whisper deletion, use right
    """
    hyps_fusion = []
    for hyp_w, hyp_z, ref in zip(hyps_whisper, hyps_zipformer, refs):
        ali = kaldialign.align(hyp_w, hyp_z, ERR)
        hyp_f = ""
        for a in ali:
            if a[0] == ERR:
                hyp_f += a[1]
            elif a[1] == ERR:
                pass
            else:
                hyp_f += a[0]
        hyps_fusion.append(hyp_f)
    return hyps_fusion


def main():
    parser = get_parser()
    args = parser.parse_args()
    # mkdir output_log_dir
    Path(args.output_log_dir).mkdir(parents=True, exist_ok=True)
    pair_logs = get_pair_filenames(args.whisper_log_dir, args.zipformer_log_dir)
    for pair in pair_logs:
        hyps_whisper, refs, wav_name = extract_hyp_ref_wavname(pair[0])
        hyps_zipformer, _, _ = extract_hyp_ref_wavname(pair[1])

        hyps_fusion = fusion_hyps_trust_substituion_insertion(
            hyps_whisper, hyps_zipformer, refs
        )

        partition_name = pair[0].split("/")[-1].split("-")[1]
        save_results(
            Path(args.output_log_dir),
            partition_name,
            {"fusion": list(zip(wav_name, refs, hyps_fusion))},
        )

        print(f"Processed {partition_name}")


if __name__ == "__main__":
    main()
