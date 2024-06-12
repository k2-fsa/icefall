#!/usr/bin/env python3
# Copyright    2024  author: Yuekai Zhang
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
import logging

from lhotse import CutSet, load_manifest_lazy


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--fixed-transcript-path",
        type=str,
        default="data/fbank/text.fix",
        help="""
        See https://github.com/wenet-e2e/WenetSpeech/discussions/54
        wget -nc https://huggingface.co/datasets/yuekai/wenetspeech_paraformer_fixed_transcript/resolve/main/text.fix
        """,
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/fbank/",
        help="Directory to store the manifest files",
    )

    parser.add_argument(
        "--training-subset",
        type=str,
        default="L",
        help="The training subset for wenetspeech.",
    )

    return parser


def load_fixed_text(fixed_text_path):
    """
    fixed text format
    X0000016287_92761015_S00001 我是徐涛
    X0000016287_92761015_S00002 狄更斯的PICK WEEK PAPERS斯
    load into a dict
    """
    fixed_text_dict = {}
    with open(fixed_text_path, "r") as f:
        for line in f:
            cut_id, text = line.strip().split(" ", 1)
            fixed_text_dict[cut_id] = text
    return fixed_text_dict


def fix_manifest(manifest, fixed_text_dict, fixed_manifest_path):
    with CutSet.open_writer(fixed_manifest_path) as manifest_writer:
        fixed_item = 0
        for i, cut in enumerate(manifest):
            if i % 10000 == 0:
                logging.info(f"Processing cut {i}, fixed {fixed_item}")
            cut_id_orgin = cut.id
            if cut_id_orgin.endswith("_sp0.9"):
                cut_id = cut_id_orgin[:-6]
            elif cut_id_orgin.endswith("_sp1.1"):
                cut_id = cut_id_orgin[:-6]
            else:
                cut_id = cut_id_orgin
            if cut_id in fixed_text_dict:
                assert (
                    len(cut.supervisions) == 1
                ), f"cut {cut_id} has {len(cut.supervisions)} supervisions"
                if cut.supervisions[0].text != fixed_text_dict[cut_id]:
                    logging.info(
                        f"Fixed text for cut {cut_id_orgin} from {cut.supervisions[0].text} to {fixed_text_dict[cut_id]}"
                    )
                    cut.supervisions[0].text = fixed_text_dict[cut_id]
                fixed_item += 1
                manifest_writer.write(cut)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    fixed_text_path = args.manifest_dir + "text.fix"
    fixed_text_dict = load_fixed_text(fixed_text_path)
    logging.info(f"Loaded {len(fixed_text_dict)} fixed texts")

    dev_manifest_path = args.manifest_dir + "cuts_DEV.jsonl.gz"
    fixed_dev_manifest_path = args.manifest_dir + "cuts_DEV_fixed.jsonl.gz"
    logging.info(f"Loading dev manifest from {dev_manifest_path}")
    cuts_dev_manifest = load_manifest_lazy(dev_manifest_path)
    fix_manifest(cuts_dev_manifest, fixed_text_dict, fixed_dev_manifest_path)
    logging.info(f"Fixed dev manifest saved to {fixed_dev_manifest_path}")

    manifest_path = args.manifest_dir + f"cuts_{args.training_subset}.jsonl.gz"
    fixed_manifest_path = (
        args.manifest_dir + f"cuts_{args.training_subset}_fixed.jsonl.gz"
    )
    logging.info(f"Loading manifest from {manifest_path}")
    cuts_manifest = load_manifest_lazy(manifest_path)
    fix_manifest(cuts_manifest, fixed_text_dict, fixed_manifest_path)
    logging.info(f"Fixed training manifest saved to {fixed_manifest_path}")


if __name__ == "__main__":
    main()
