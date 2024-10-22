#!/usr/bin/env python3
#
# Copyright      2023 Xiaomi Corporation     (Author: Zengwei Yao,
#                                                     Zengrui Jin,)
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
This script performs model inference on test set.

Usage:
./vits/infer.py \
    --epoch 1000 \
    --exp-dir ./vits/exp \
    --max-duration 500
"""


import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import k2
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from lhotse.features.io import KaldiReader
from tokenizer import Tokenizer
from train import get_model, get_params
from tts_datamodule import LibrittsTtsDataModule

from icefall.checkpoint import load_checkpoint
from icefall.utils import AttributeDict, setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="vits/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    return parser


def infer_dataset(
    dl: torch.utils.data.DataLoader,
    subset: str,
    params: AttributeDict,
    model: nn.Module,
    tokenizer: Tokenizer,
    speaker_map: KaldiReader,
) -> None:
    """Decode dataset.
    The ground-truth and generated audio pairs will be saved to `params.save_wav_dir`.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      tokenizer:
        Used to convert text to phonemes.
    """

    #  Background worker save audios to disk.
    def _save_worker(
        subset: str,
        batch_size: int,
        cut_ids: List[str],
        audio: torch.Tensor,
        audio_pred: torch.Tensor,
        audio_lens: List[int],
        audio_lens_pred: List[int],
    ):
        for i in range(batch_size):
            torchaudio.save(
                str(params.save_wav_dir / subset / f"{cut_ids[i]}_gt.wav"),
                audio[i : i + 1, : audio_lens[i]],
                sample_rate=params.sampling_rate,
            )
            torchaudio.save(
                str(params.save_wav_dir / subset / f"{cut_ids[i]}_pred.wav"),
                audio_pred[i : i + 1, : audio_lens_pred[i]],
                sample_rate=params.sampling_rate,
            )

    device = next(model.parameters()).device
    num_cuts = 0
    log_interval = 5

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for batch_idx, batch in enumerate(dl):
            batch_size = len(batch["tokens"])

            tokens = batch["tokens"]
            tokens = tokenizer.tokens_to_token_ids(
                tokens, intersperse_blank=True, add_sos=True, add_eos=True
            )
            tokens = k2.RaggedTensor(tokens)
            row_splits = tokens.shape.row_splits(1)
            tokens_lens = row_splits[1:] - row_splits[:-1]
            tokens = tokens.to(device)
            tokens_lens = tokens_lens.to(device)
            # tensor of shape (B, T)
            tokens = tokens.pad(mode="constant", padding_value=tokenizer.pad_id)

            audio = batch["audio"]
            audio_lens = batch["audio_lens"].tolist()
            cut_ids = [cut.id for cut in batch["cut"]]
            sids = ["_".join(cut_id.split("_")[:2]) for cut_id in cut_ids]
            spembs = (
                torch.Tensor(np.array([speaker_map.read(sid) for sid in sids]))
                .squeeze(1)
                .to(device)
            )

            audio_pred, _, durations = model.inference_batch(
                text=tokens,
                text_lengths=tokens_lens,
                spembs=spembs,
            )
            audio_pred = audio_pred.detach().cpu()
            # convert to samples
            audio_lens_pred = (
                (durations.sum(1) * params.frame_shift).to(dtype=torch.int64).tolist()
            )

            futures.append(
                executor.submit(
                    _save_worker,
                    subset,
                    batch_size,
                    cut_ids,
                    audio,
                    audio_pred,
                    audio_lens,
                    audio_lens_pred,
                )
            )

            num_cuts += batch_size

            if batch_idx % log_interval == 0:
                batch_str = f"{batch_idx}/{num_batches}"

                logging.info(
                    f"batch {batch_str}, cuts processed until now is {num_cuts}"
                )
        # return results
        for f in futures:
            f.result()


@torch.no_grad()
def main():
    parser = get_parser()
    LibrittsTtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.suffix = f"epoch-{params.epoch}"

    params.res_dir = params.exp_dir / "infer" / params.suffix
    params.save_wav_dir = params.res_dir / "wav"
    params.save_wav_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(f"{params.res_dir}/log-infer-{params.suffix}")
    logging.info("Infer started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    tokenizer = Tokenizer(params.tokens)
    params.blank_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size

    # we need cut ids to display recognition results.
    args.return_cuts = True
    libritts = LibrittsTtsDataModule(args)

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    model.to(device)
    model.eval()

    num_param_g = sum([p.numel() for p in model.generator.parameters()])
    logging.info(f"Number of parameters in generator: {num_param_g}")
    num_param_d = sum([p.numel() for p in model.discriminator.parameters()])
    logging.info(f"Number of parameters in discriminator: {num_param_d}")
    logging.info(f"Total number of parameters: {num_param_g + num_param_d}")

    test_clean_cuts = libritts.test_clean_cuts()
    test_clean_speaker_map = libritts.test_clean_xvector()
    test_clean_dl = libritts.test_dataloaders(test_clean_cuts)

    dev_clean_cuts = libritts.dev_clean_cuts()
    dev_clean_speaker_map = libritts.dev_clean_xvector()
    dev_clean_dl = libritts.dev_dataloaders(dev_clean_cuts)

    infer_sets = {
        "test-clean": (test_clean_dl, test_clean_speaker_map),
        "dev-clean": (dev_clean_dl, dev_clean_speaker_map),
    }

    for subset, data in infer_sets.items():
        save_wav_dir = params.res_dir / "wav" / subset
        save_wav_dir.mkdir(parents=True, exist_ok=True)
        dl, speaker_map = data

        logging.info(f"Processing {subset} set, saving to {save_wav_dir}")

        infer_dataset(
            dl=dl,
            subset=subset,
            params=params,
            model=model,
            tokenizer=tokenizer,
            speaker_map=speaker_map,
        )

    logging.info(f"Wav files are saved to {params.save_wav_dir}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
