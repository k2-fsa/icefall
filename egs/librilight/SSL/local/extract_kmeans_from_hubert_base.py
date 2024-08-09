#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Yifan Yang)
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
from pathlib import Path
from typing import Optional

import fairseq
import joblib
import numpy as np
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.utils import fastcopy
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        default="small",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="download/hubert_base_ls960.pt",
    )

    parser.add_argument(
        "--kmeans-model-path",
        type=str,
        default="download/hubert_base_ls960_L9_km500.model",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (inclusive).",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (exclusive).",
    )

    return parser.parse_args()


def extract_and_save_one_cuts(
    raw_cuts_path, cuts_path, model, vad_model, apply_kmeans, do_normalize, device
):
    logging.info(f"Loading {raw_cuts_path}")
    cut_set = CutSet.from_file(raw_cuts_path)

    logging.info("Extracting kmeans")
    cuts = []
    for cut in tqdm(cut_set):
        assert cut.sampling_rate == 16000, f"{cut.sampling_rate}"
        audio = cut.load_audio()

        if audio.shape[-1] > 64 * 16000:
            timestamps = get_speech_timestamps(audio, vad_model)
            offsets = [i["start"] for i in timestamps]
            audios = [audio[:, i["start"] : i["end"]] for i in timestamps]
            logging.info(f"Trim audio {cut.id} into {len(audios)} segments")
        else:
            offsets = [0]
            audios = [audio]

        seq = 0
        for audio, offset in zip(audios, offsets):
            x = torch.from_numpy(audio).float().to(device)

            with torch.no_grad():
                if do_normalize:
                    x = torch.nn.functional.layer_norm(x, x.shape)

                feature, _ = model.extract_features(
                    source=x,
                    padding_mask=None,
                    mask=False,
                    output_layer=9,
                )
                feature = feature.squeeze(0)

            kmeans = " ".join(map(str, apply_kmeans(feature).tolist()))

            supervision_segment = fastcopy(
                cut.supervisions[0],
                id=f"{cut.id}-{seq}",
                start=0.0,
                duration=audio.shape[-1] / 16000,
            )
            cut_with_kmeans = fastcopy(
                cut,
                id=f"{cut.id}-{seq}",
                start=cut.start + offset / 16000,
                duration=audio.shape[-1] / 16000,
                supervisions=[supervision_segment],
                custom={"kmeans": kmeans},
            )
            cuts.append(cut_with_kmeans)

            seq += 1

    cuts = CutSet(cuts)

    logging.info(f"Saving to {cuts_path}")
    cuts.to_file(cuts_path)


def extract_kmeans(args):
    assert args.subset in ("small", "medium", "large"), f"{args.subset}"

    output_dir = (
        f"data/kmeans/{args.subset}_split" if args.subset != "small" else "data/kmeans"
    )
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    prefix = "librilight"

    vad_model = load_silero_vad()
    apply_kmeans = ApplyKmeans(args.kmeans_model_path)
    model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.model_path]
    )
    model = model[0].eval().to(device)
    do_normalize = task.cfg.normalize

    if args.subset == "small":
        cuts_path = output_dir / f"{prefix}_cuts_{args.subset}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            return

        raw_cuts_path = output_dir / f"{prefix}_cuts_{args.subset}_raw.jsonl.gz"
        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            return

        extract_and_save_one_cuts(
            raw_cuts_path,
            cuts_path,
            model,
            vad_model,
            apply_kmeans,
            do_normalize,
            device,
        )
    else:
        num_digits = 8  # num_digits is fixed by lhotse split-lazy
        start = args.start
        stop = args.stop
        assert stop > start, "stop must be larger than start!"

        for i in range(start, stop):
            idx = f"{i}".zfill(num_digits)
            logging.info(f"Processing {idx}/{stop - 1}")

            cuts_path = output_dir / f"{prefix}_cuts_{args.subset}.{idx}.jsonl.gz"
            if cuts_path.is_file():
                logging.info(f"{cuts_path} exists - skipping")
                continue

            raw_cuts_path = (
                output_dir / f"{prefix}_cuts_{args.subset}_raw.{idx}.jsonl.gz"
            )
            if not raw_cuts_path.is_file():
                logging.info(f"{raw_cuts_path} does not exist - skipping it")
                continue

            extract_and_save_one_cuts(
                raw_cuts_path,
                cuts_path,
                model,
                vad_model,
                apply_kmeans,
                do_normalize,
                device,
            )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    extract_kmeans(args)
