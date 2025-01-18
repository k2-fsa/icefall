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
import math
import os
from pathlib import Path
from typing import Optional

import fairseq
import joblib
import numpy as np
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.utils import fastcopy
from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


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
        default="download/hubert_base_ls960_L9_km500.bin",
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

    parser.add_argument(
        "--window-duration",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--shift-duration",
        type=float,
        default=250.0,
    )

    return parser.parse_args()


@torch.no_grad()
def extract_and_save_one_cuts(
    raw_cuts_path,
    cuts_path,
    model,
    apply_kmeans,
    do_normalize,
    window_duration,
    shift_duration,
):
    logging.info(f"Loading {raw_cuts_path}")
    cut_set = CutSet.from_file(raw_cuts_path)

    logging.info("Extracting kmeans")
    cuts = []

    assert window_duration >= shift_duration
    window_size = int(window_duration * 16000)
    shift_size = int(shift_duration * 16000)
    overlap_size = window_size - shift_size
    out_overlap_size = get_out_length(overlap_size)

    for cut in tqdm(cut_set):
        assert cut.sampling_rate == 16000, f"Sampling rate: {cut.sampling_rate}"

        audio = cut.load_audio()

        T = audio.shape[1]
        start = 0
        kmeans = []
        while start < T:
            real_window_size = min(window_size, T - start)
            audio_window = audio[:, start : start + real_window_size]

            x = (
                torch.from_numpy(audio_window)
                .float()
                .to(next(model.parameters()).device)
            )
            if do_normalize:
                x = torch.nn.functional.layer_norm(x, x.shape)

            feature, _ = model.extract_features(
                source=x,
                padding_mask=None,
                mask=False,
                output_layer=9,
            )
            feature = feature.squeeze(0)

            current_kmeans = apply_kmeans(feature).tolist()

            if start == 0:
                kmeans.extend(current_kmeans)
            else:
                kmeans.extend(current_kmeans[out_overlap_size:])

            if T - start <= window_size:
                break

            start += shift_size

        kmeans = " ".join(map(str, kmeans))

        cut_with_kmeans = fastcopy(
            cut,
            custom={"kmeans": kmeans},
        )
        cuts.append(cut_with_kmeans)

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

    apply_kmeans = ApplyKmeans(args.kmeans_model_path)
    model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.model_path]
    )
    model = model[0].eval().to(device)
    do_normalize = task.cfg.normalize

    window_duration = args.window_duration
    shift_duration = args.shift_duration

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
            apply_kmeans,
            do_normalize,
            window_duration,
            shift_duration,
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
                apply_kmeans,
                do_normalize,
                window_duration,
                shift_duration,
            )


def get_out_length(T):
    conv_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    for i, (out_channels, kernel_size, stride) in enumerate(conv_layers):
        T = math.floor((T - kernel_size) / stride) + 1

    return max(0, T)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    extract_kmeans(args)
