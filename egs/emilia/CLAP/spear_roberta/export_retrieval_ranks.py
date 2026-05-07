#!/usr/bin/env python3
# Copyright      2025  Yifan Yang
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
import json
import os
from pathlib import Path

from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="spear_roberta/exp",
        help="The experiment dir",
    )

    return parser


def export_audio_to_text(details, output_dir):
    """
    audio_to_text_ranks:
      audio -> [text0, text1, ...]
    """
    for idx, (audio_path, texts) in enumerate(details.items()):
        item_dir = output_dir / str(idx)
        item_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)
        os.symlink(audio_path.resolve(), item_dir / audio_path.name)

        for rank, text in enumerate(texts):
            with open(item_dir / f"{rank}.txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")


def export_text_to_audio(details, output_dir):
    """
    text_to_audio_ranks:
      text -> [audio0, audio1, ...]
    """

    for idx, (text, audio_paths) in enumerate(details.items()):
        item_dir = output_dir / str(idx)
        item_dir.mkdir(parents=True, exist_ok=True)

        with open(item_dir / "text.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")

        for rank, audio_path in enumerate(audio_paths):
            audio_path = audio_path.replace("GT# ", "")
            audio_path = Path(audio_path)
            os.symlink(audio_path.resolve(), item_dir / f"{rank}{audio_path.suffix}")


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    args.res_dir = args.exp_dir / "speech-text-retrieval"

    if args.iter > 0:
        args.suffix = f"iter-{args.iter}-avg-{args.avg}"
    else:
        args.suffix = f"epoch-{args.epoch}-avg-{args.avg}"

    if args.use_averaged_model:
        args.suffix += "-use-averaged-model"

    with open(f"{args.res_dir}/details-decode-{args.suffix}", encoding="utf-8") as f:
        details = json.load(f)

    export_audio_to_text(
        details["audio_to_text_ranks"],
        args.res_dir / args.suffix / "audio_to_text_ranks",
    )

    export_text_to_audio(
        details["text_to_audio_ranks"],
        args.res_dir / args.suffix / "text_to_audio_ranks",
    )


if __name__ == "__main__":
    main()
