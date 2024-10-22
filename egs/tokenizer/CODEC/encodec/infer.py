#!/usr/bin/env python3
#
# Copyright      2024 The Chinese University of HK   (Author: Zengrui Jin)
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
./codec/infer.py \
    --epoch 300 \
    --exp-dir ./codec/exp \
    --max-duration 500
"""


import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from codec_datamodule import LibriTTSCodecDataModule
from pesq import pesq
from pystoi import stoi
from scipy import signal
from torch import nn
from train import get_model, get_params

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
        default="encodec/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--target-bw",
        type=float,
        default=24,
        help="The target bandwidth for the generator",
    )

    return parser


# implementation from https://github.com/yangdongchao/AcademiCodec/blob/master/academicodec/models/encodec/test.py
def remove_encodec_weight_norm(model) -> None:
    from modules import SConv1d
    from modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


def compute_pesq(ref_wav: np.ndarray, gen_wav: np.ndarray) -> float:
    """Compute PESQ score between reference and generated audio."""
    DEFAULT_SAMPLING_RATE = 16000
    ref = signal.resample(ref_wav, DEFAULT_SAMPLING_RATE)
    deg = signal.resample(gen_wav, DEFAULT_SAMPLING_RATE)
    return pesq(fs=DEFAULT_SAMPLING_RATE, ref=ref, deg=deg, mode="wb")


def compute_stoi(ref_wav: np.ndarray, gen_wav: np.ndarray, sampling_rate: int) -> float:
    """Compute STOI score between reference and generated audio."""
    return stoi(x=ref_wav, y=gen_wav, fs_sig=sampling_rate, extended=False)


def infer_dataset(
    dl: torch.utils.data.DataLoader,
    subset: str,
    params: AttributeDict,
    model: nn.Module,
) -> Tuple[float, float]:
    """Decode dataset.
    The ground-truth and generated audio pairs will be saved to `params.save_wav_dir`.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      subset:
        The name of the subset.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.

    Returns:
        The average PESQ and STOI scores.
    """

    #  Background worker save audios to disk.
    def _save_worker(
        subset: str,
        batch_size: int,
        cut_ids: List[str],
        audio: torch.Tensor,
        audio_pred: torch.Tensor,
        audio_lens: List[int],
    ):
        for i in range(batch_size):
            torchaudio.save(
                str(params.save_wav_dir / subset / f"{cut_ids[i]}_gt.wav"),
                audio[i : i + 1, : audio_lens[i]],
                sample_rate=params.sampling_rate,
            )
            torchaudio.save(
                str(params.save_wav_dir / subset / f"{cut_ids[i]}_recon.wav"),
                audio_pred[i : i + 1, : audio_lens[i]],
                sample_rate=params.sampling_rate,
            )

    device = next(model.parameters()).device
    num_cuts = 0
    log_interval = 5

    pesq_wb_scores = []
    stoi_scores = []

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for batch_idx, batch in enumerate(dl):
            batch_size = len(batch["audio"])

            audios = batch["audio"]
            audio_lens = batch["audio_lens"].tolist()
            cut_ids = [cut.id for cut in batch["cut"]]

            codes, audio_hats = model.inference(
                audios.to(device), target_bw=params.target_bw
            )
            audio_hats = audio_hats.squeeze(1).cpu()

            for cut_id, audio, audio_hat, audio_len in zip(
                cut_ids, audios, audio_hats, audio_lens
            ):
                try:
                    pesq_wb = compute_pesq(
                        ref_wav=audio[:audio_len].numpy(),
                        gen_wav=audio_hat[:audio_len].numpy(),
                    )
                    pesq_wb_scores.append(pesq_wb)
                except Exception as e:
                    logging.error(f"Error while computing PESQ for cut {cut_id}: {e}")

                stoi_score = compute_stoi(
                    ref_wav=audio[:audio_len].numpy(),
                    gen_wav=audio_hat[:audio_len].numpy(),
                    sampling_rate=params.sampling_rate,
                )
                stoi_scores.append(stoi_score)

            futures.append(
                executor.submit(
                    _save_worker,
                    subset,
                    batch_size,
                    cut_ids,
                    audios,
                    audio_hats,
                    audio_lens,
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
    return mean(pesq_wb_scores), mean(stoi_scores)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriTTSCodecDataModule.add_arguments(parser)
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

    # we need cut ids to display results of both constructed and ground-truth audio
    args.return_cuts = True
    libritts = LibriTTSCodecDataModule(args)

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    remove_encodec_weight_norm(model)

    model.to(device)
    model.eval()

    encoder = model.encoder
    decoder = model.decoder
    quantizer = model.quantizer
    multi_scale_discriminator = model.multi_scale_discriminator
    multi_period_discriminator = model.multi_period_discriminator
    multi_scale_stft_discriminator = model.multi_scale_stft_discriminator

    num_param_e = sum([p.numel() for p in encoder.parameters()])
    logging.info(f"Number of parameters in encoder: {num_param_e}")
    num_param_d = sum([p.numel() for p in decoder.parameters()])
    logging.info(f"Number of parameters in decoder: {num_param_d}")
    num_param_q = sum([p.numel() for p in quantizer.parameters()])
    logging.info(f"Number of parameters in quantizer: {num_param_q}")
    num_param_ds = (
        sum([p.numel() for p in multi_scale_discriminator.parameters()])
        if multi_scale_discriminator is not None
        else 0
    )
    logging.info(f"Number of parameters in multi_scale_discriminator: {num_param_ds}")
    num_param_dp = (
        sum([p.numel() for p in multi_period_discriminator.parameters()])
        if multi_period_discriminator is not None
        else 0
    )
    logging.info(f"Number of parameters in multi_period_discriminator: {num_param_dp}")
    num_param_dstft = sum(
        [p.numel() for p in multi_scale_stft_discriminator.parameters()]
    )
    logging.info(
        f"Number of parameters in multi_scale_stft_discriminator: {num_param_dstft}"
    )
    logging.info(
        f"Total number of parameters: {num_param_e + num_param_d + num_param_q + num_param_ds + num_param_dp + num_param_dstft}"
    )

    test_clean_cuts = libritts.test_clean_cuts()
    test_clean = libritts.test_dataloaders(test_clean_cuts)

    test_other_cuts = libritts.test_other_cuts()
    test_other = libritts.test_dataloaders(test_other_cuts)

    dev_clean_cuts = libritts.dev_clean_cuts()
    dev_clean = libritts.valid_dataloaders(dev_clean_cuts)

    dev_other_cuts = libritts.dev_other_cuts()
    dev_other = libritts.valid_dataloaders(dev_other_cuts)

    infer_sets = {
        "test-clean": test_clean,
        "test-other": test_other,
        "dev-clean": dev_clean,
        "dev-other": dev_other,
    }

    for subset, dl in infer_sets.items():
        save_wav_dir = params.res_dir / "wav" / subset
        save_wav_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing {subset} set, saving to {save_wav_dir}")

        pesq_wb, stoi = infer_dataset(
            dl=dl,
            subset=subset,
            params=params,
            model=model,
        )
        logging.info(f"{subset}: PESQ-WB: {pesq_wb:.4f}, STOI: {stoi:.4f}")

    logging.info(f"Wav files are saved to {params.save_wav_dir}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
