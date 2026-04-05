#!/usr/bin/env python3
# Copyright    2025  author: Yuekai Zhang
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
import argparse
import gzip
import json
import logging

import s3tokenizer
from lhotse import CutSet, load_manifest_lazy
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/fbank",
        help="Directory to store the manifest files",
    )

    parser.add_argument(
        "--jsonl-prefix",
        type=str,
        default="wenetspeech4tts_cuts_valid",
        help="The training subset for wenetspeech.",
    )

    parser.add_argument(
        "--tokens-path",
        type=str,
        default="./s3_tokens_valid/wenetspeech4tts_valid.json",
        help="json file containing the speech tokens",
    )

    return parser


def get_speech_tokens(tokens_path):
    id2tokens = {}
    with open(tokens_path, "r") as fin:
        for line in fin:
            line = json.loads(line)
            id2tokens[line["key"]] = " ".join(map(str, line["code"]))
    return id2tokens


def attach_manifest(manifest, fixed_manifest_path, id2tokens):
    with CutSet.open_writer(fixed_manifest_path) as manifest_writer:
        fixed_item = 0
        for i, cut in enumerate(tqdm(manifest)):
            cut_id = cut.supervisions[0].id
            if cut_id in id2tokens:
                code = id2tokens[cut_id]
                cut.supervisions[0].custom = {
                    **cut.supervisions[0].custom,
                    **{"speech_tokens": code},
                }
            else:
                print(f"cut_id {cut_id} not in id2tokens")
            fixed_item += 1
            manifest_writer.write(cut)
    logging.info(f"Fixed {fixed_item} items in the manifest")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    manifest_path = args.manifest_dir + "/" + f"{args.jsonl_prefix}.jsonl.gz"
    attached_manifest_path = (
        args.manifest_dir + "/" + f"{args.jsonl_prefix}_attached_cosyvoice_v2.jsonl.gz"
    )
    logging.info(f"Loading manifest from {manifest_path}")
    cuts_manifest = load_manifest_lazy(manifest_path)
    logging.info(f"Loading manifest from {manifest_path} done")
    id2tokens = get_speech_tokens(args.tokens_path)
    logging.info(f"Loaded id2tokens with {len(id2tokens)} entries")

    attach_manifest(cuts_manifest, attached_manifest_path, id2tokens)
    logging.info(
        f"Manifest with speech tokens attached is saved to {attached_manifest_path}"
    )


if __name__ == "__main__":
    main()
