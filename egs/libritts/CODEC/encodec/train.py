#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.                  (Author: Zengwei Yao)
#                   2024 The Chinese University of HK   (Author: Zengrui Jin)
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
import itertools
import logging
import math
import random
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from codec_datamodule import LibriTTSCodecDataModule
from encodec import Encodec
from lhotse.utils import fix_random_seed
from scheduler import WarmupCosineLrScheduler
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from utils import MetricsTracker, save_checkpoint

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, setup_logger, str2bool

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to generate for tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=500,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="encodec/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lr", type=float, default=3.0e-4, help="The base learning rate."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=5,
        help="""Save checkpoint after processing this number of epochs"
        periodically. We save checkpoint to exp-dir/ whenever
        params.cur_epoch % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/epoch-{params.cur_epoch}.pt'.
        Since it will take around 1000 epochs, we suggest using a large
        save_every_n to save disk space.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0


    """
    params = AttributeDict(
        {
            # training params
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": -1,  # 0
            "log_interval": 50,
            "valid_interval": 200,
            "env_info": get_env_info(),
            "sampling_rate": 24000,
            "audio_normalization": False,
            "lambda_adv": 3.0,  # loss scaling coefficient for adversarial loss
            "lambda_wav": 0.1,  # loss scaling coefficient for waveform loss
            "lambda_feat": 4.0,  # loss scaling coefficient for feat loss
            "lambda_rec": 1.0,  # loss scaling coefficient for reconstruction loss
            "lambda_com": 1000.0,  # loss scaling coefficient for commitment loss
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict, model: nn.Module
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(filename, model=model)

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def get_model(params: AttributeDict) -> nn.Module:
    """Get the model based on the configuration."""

    from discriminators import (
        MultiPeriodDiscriminator,
        MultiScaleDiscriminator,
        MultiScaleSTFTDiscriminator,
    )
    from modules.seanet import SEANetDecoder, SEANetEncoder
    from quantization import ResidualVectorQuantizer

    # generator_params = {
    #     "generator_n_filters": 32,
    #     "dimension": 512,
    #     "ratios": [2, 2, 2, 4],
    #     "target_bandwidths": [7.5, 15],
    #     "bins": 1024,
    # }
    # discriminator_params = {
    #     "stft_discriminator_n_filters": 32,
    #     "discriminator_epoch_start": 5,
    # }
    # inference_params = {
    #     "target_bw": 7.5,
    # }

    generator_params = {
        "generator_n_filters": 32,
        "dimension": 512,
        "ratios": [8, 5, 4, 2],
        "target_bandwidths": [1.5, 3, 6, 12, 24],
        "bins": 1024,
    }
    discriminator_params = {
        "stft_discriminator_n_filters": 32,
        "discriminator_epoch_start": 5,
        "n_ffts": [1024, 2048, 512],
        "hop_lengths": [256, 512, 128],
        "win_lengths": [1024, 2048, 512],
    }
    inference_params = {
        "target_bw": 6,
    }

    params.update(generator_params)
    params.update(discriminator_params)
    params.update(inference_params)

    hop_length = np.prod(params.ratios)
    n_q = int(
        1000
        * params.target_bandwidths[-1]
        // (math.ceil(params.sampling_rate / hop_length) * 10)
    )

    encoder = SEANetEncoder(
        n_filters=params.generator_n_filters,
        dimension=params.dimension,
        ratios=params.ratios,
    )
    decoder = SEANetDecoder(
        n_filters=params.generator_n_filters,
        dimension=params.dimension,
        ratios=params.ratios,
    )
    quantizer = ResidualVectorQuantizer(
        dimension=params.dimension, n_q=n_q, bins=params.bins
    )

    model = Encodec(
        params=params,
        sampling_rate=params.sampling_rate,
        target_bandwidths=params.target_bandwidths,
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        multi_scale_discriminator=None,
        multi_period_discriminator=None,
        multi_scale_stft_discriminator=MultiScaleSTFTDiscriminator(
            n_filters=params.stft_discriminator_n_filters,
            n_ffts=params.n_ffts,
            hop_lengths=params.hop_lengths,
            win_lengths=params.win_lengths,
        ),
    )
    return model


def prepare_input(
    params: AttributeDict,
    batch: dict,
    device: torch.device,
    is_training: bool = True,
):
    """Parse batch data"""
    audio = batch["audio"].to(device, memory_format=torch.contiguous_format)
    features = batch["features"].to(device, memory_format=torch.contiguous_format)
    audio_lens = batch["audio_lens"].to(device)
    features_lens = batch["features_lens"].to(device)

    if is_training:
        audio_dims = audio.size(-1)
        start_idx = random.randint(0, max(0, audio_dims - params.sampling_rate))
        audio = audio[:, start_idx : params.sampling_rate + start_idx]
    else:
        # NOTE(zengrui): a very coarse setup
        audio = audio[
            :, params.sampling_rate : params.sampling_rate + params.sampling_rate
        ]

    if params.audio_normalization:
        mean = audio.mean(dim=-1, keepdim=True)
        std = audio.std(dim=-1, keepdim=True)
        audio = (audio - mean) / (std + 1e-7)

    return audio, audio_lens, features, features_lens


def train_discriminator(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer_g: Optimizer,
    optimizer_d: Optimizer,
    scheduler_g: LRSchedulerType,
    scheduler_d: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model to be trained.
      optimizer_g:
        The optimizer for generator.
      optimizer_d:
        The optimizer for discriminator.
      scheduler_g:
        The learning rate scheduler for generator, we call step() every epoch.
      scheduler_d:
        The learning rate scheduler for discriminator, we call step() every epoch.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            params=params,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1

        batch_size = len(batch["audio"])
        (
            audio,
            audio_lens,
            _,
            _,
        ) = prepare_input(params, batch, device)

        loss_info = MetricsTracker()
        loss_info["samples"] = batch_size

        try:
            with autocast(enabled=params.use_fp16):
                d_weight = train_discriminator(
                    params.lambda_adv,
                    params.cur_epoch,
                    threshold=params.discriminator_epoch_start,
                )
                # forward discriminator
                (
                    disc_stft_real_adv_loss,
                    disc_stft_fake_adv_loss,
                    disc_period_real_adv_loss,
                    disc_period_fake_adv_loss,
                    disc_scale_real_adv_loss,
                    disc_scale_fake_adv_loss,
                    stats_d,
                ) = model(
                    speech=audio,
                    speech_lengths=audio_lens,
                    return_sample=False,
                    forward_generator=False,
                )
                disc_loss = (
                    disc_stft_real_adv_loss
                    + disc_stft_fake_adv_loss
                    + disc_period_real_adv_loss
                    + disc_period_fake_adv_loss
                    + disc_scale_real_adv_loss
                    + disc_scale_fake_adv_loss
                ) * d_weight
            for k, v in stats_d.items():
                loss_info[k] = v * batch_size
            # update discriminator
            optimizer_d.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(optimizer_d)

            with autocast(enabled=params.use_fp16):
                g_weight = train_discriminator(
                    params.lambda_adv,
                    params.cur_epoch,
                    threshold=params.discriminator_epoch_start,
                )
                # forward generator
                (
                    commit_loss,
                    gen_stft_adv_loss,
                    gen_period_adv_loss,
                    gen_scale_adv_loss,
                    feature_stft_loss,
                    feature_period_loss,
                    feature_scale_loss,
                    wav_reconstruction_loss,
                    mel_reconstruction_loss,
                    stats_g,
                ) = model(
                    speech=audio,
                    speech_lengths=audio_lens,
                    forward_generator=True,
                    return_sample=params.batch_idx_train % params.log_interval == 0,
                )
                gen_adv_loss = (
                    gen_stft_adv_loss + gen_period_adv_loss + gen_scale_adv_loss
                ) * g_weight
                feature_loss = (
                    feature_stft_loss + feature_period_loss + feature_scale_loss
                )
                reconstruction_loss = (
                    params.lambda_wav * wav_reconstruction_loss
                    + params.lambda_rec * mel_reconstruction_loss
                )
                gen_loss = (
                    gen_adv_loss
                    + reconstruction_loss
                    + params.lambda_feat * feature_loss
                    + params.lambda_com * commit_loss
                )
            loss_info["generator_loss"] = gen_loss
            for k, v in stats_g.items():
                if "returned_sample" not in k:
                    loss_info[k] = v * batch_size
            # update generator
            optimizer_g.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # summary stats
            tot_loss = tot_loss + loss_info
        except:  # noqa
            save_bad_model()
            raise

        # step per iteration
        scheduler_g.step()
        scheduler_d.step()

        if params.print_diagnostics and batch_idx == 5:
            return

        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (
                cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr_g = max(scheduler_g.get_last_lr())
            cur_lr_d = max(scheduler_d.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr_g: {cur_lr_g:.2e}, cur_lr_d: {cur_lr_d:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate_g", cur_lr_g, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/learning_rate_d", cur_lr_d, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )
                if "returned_sample" in stats_g:
                    # speech_hat_, speech_, mel_hat_, mel_ = stats_g["returned_sample"]
                    speech_hat_, speech_, _, _ = stats_g["returned_sample"]

                    speech_hat_i = speech_hat_[0]
                    speech_i = speech_[0]
                    if speech_hat_i.dim() > 1:
                        speech_hat_i = speech_hat_i.squeeze(0)
                        speech_i = speech_i.squeeze(0)
                    tb_writer.add_audio(
                        f"train/speech_hat_",
                        speech_hat_i,
                        params.batch_idx_train,
                        params.sampling_rate,
                    )
                    tb_writer.add_audio(
                        f"train/speech_",
                        speech_i,
                        params.batch_idx_train,
                        params.sampling_rate,
                    )
                    # tb_writer.add_image(
                    #     "train/mel_hat_",
                    #     plot_feature(mel_hat_),
                    #     params.batch_idx_train,
                    #     dataformats="HWC",
                    # )
                    # tb_writer.add_image(
                    #     "train/mel_",
                    #     plot_feature(mel_),
                    #     params.batch_idx_train,
                    #     dataformats="HWC",
                    # )

        if (
            params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info, (speech_hat, speech) = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                rank=rank,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None and rank == 0:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )
                for index in range(params.num_samples):  # 3
                    speech_hat_i = speech_hat[index]
                    speech_i = speech[index]
                    if speech_hat_i.dim() > 1:
                        speech_hat_i = speech_hat_i.squeeze(0)
                        speech_i = speech_i.squeeze(0)
                    tb_writer.add_audio(
                        f"train/valid_speech_hat_{index}",
                        speech_hat_i,
                        params.batch_idx_train,
                        params.sampling_rate,
                    )
                    tb_writer.add_audio(
                        f"train/valid_speech_{index}",
                        speech_i,
                        params.batch_idx_train,
                        params.sampling_rate,
                    )

    loss_value = tot_loss["generator_loss"] / tot_loss["samples"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[MetricsTracker, Tuple[np.ndarray, np.ndarray]]:
    """Run the validation process."""
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()
    returned_sample = (None, None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            batch_size = len(batch["audio"])
            (
                audio,
                audio_lens,
                _,
                _,
            ) = prepare_input(params, batch, device, is_training=False)

            loss_info = MetricsTracker()
            loss_info["samples"] = batch_size

            d_weight = train_discriminator(
                params.lambda_adv,
                params.cur_epoch,
                threshold=params.discriminator_epoch_start,
            )

            # forward discriminator
            (
                disc_stft_real_adv_loss,
                disc_stft_fake_adv_loss,
                disc_period_real_adv_loss,
                disc_period_fake_adv_loss,
                disc_scale_real_adv_loss,
                disc_scale_fake_adv_loss,
                stats_d,
            ) = model(
                speech=audio,
                speech_lengths=audio_lens,
                return_sample=False,
                forward_generator=False,
            )
            disc_loss = (
                disc_stft_real_adv_loss
                + disc_stft_fake_adv_loss
                + disc_period_real_adv_loss
                + disc_period_fake_adv_loss
                + disc_scale_real_adv_loss
                + disc_scale_fake_adv_loss
            ) * d_weight
            assert disc_loss.requires_grad is False
            loss_info["discriminator_loss"] = disc_loss
            for k, v in stats_d.items():
                loss_info[k] = v * batch_size

            g_weight = train_discriminator(
                params.lambda_adv,
                params.cur_epoch,
                threshold=params.discriminator_epoch_start,
            )
            # forward generator
            (
                commit_loss,
                gen_stft_adv_loss,
                gen_period_adv_loss,
                gen_scale_adv_loss,
                feature_stft_loss,
                feature_period_loss,
                feature_scale_loss,
                wav_reconstruction_loss,
                mel_reconstruction_loss,
                stats_g,
            ) = model(
                speech=audio,
                speech_lengths=audio_lens,
                forward_generator=True,
                return_sample=False,
            )
            gen_adv_loss = (
                gen_stft_adv_loss + gen_period_adv_loss + gen_scale_adv_loss
            ) * g_weight
            feature_loss = feature_stft_loss + feature_period_loss + feature_scale_loss
            reconstruction_loss = (
                params.lambda_wav * wav_reconstruction_loss
                + params.lambda_rec * mel_reconstruction_loss
            )
            gen_loss = (
                gen_adv_loss
                + reconstruction_loss
                + params.lambda_feat * feature_loss
                + params.lambda_com * commit_loss
            )
            assert gen_loss.requires_grad is False
            loss_info["generator_loss"] = gen_loss
            for k, v in stats_g.items():
                if "returned_sample" not in k:
                    loss_info[k] = v * batch_size

            # summary stats
            tot_loss = tot_loss + loss_info

            # infer for first batch:
            if batch_idx == 0 and rank == 0:
                inner_model = model.module if isinstance(model, DDP) else model
                _, audio_hat = inner_model.inference(
                    x=audio, target_bw=params.target_bw
                )
                returned_sample = (audio_hat, audio)

    if world_size > 1:
        tot_loss.reduce(device)

    loss_value = tot_loss["generator_loss"] / tot_loss["samples"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss, returned_sample


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        (
            audio,
            audio_lens,
            _,
            _,
        ) = prepare_input(params, batch, device)
        try:
            # for discriminator
            with autocast(enabled=params.use_fp16):
                (
                    disc_stft_real_adv_loss,
                    disc_stft_fake_adv_loss,
                    disc_period_real_adv_loss,
                    disc_period_fake_adv_loss,
                    disc_scale_real_adv_loss,
                    disc_scale_fake_adv_loss,
                    stats_d,
                ) = model(
                    speech=audio,
                    speech_lengths=audio_lens,
                    return_sample=False,
                    forward_generator=False,
                )
            loss_d = (
                disc_stft_real_adv_loss
                + disc_stft_fake_adv_loss
                + disc_period_real_adv_loss
                + disc_period_fake_adv_loss
                + disc_scale_real_adv_loss
                + disc_scale_fake_adv_loss
            ) * train_discriminator(
                params.lambda_adv,
                params.cur_epoch,
                threshold=params.discriminator_train_start,
            )
            optimizer_d.zero_grad()
            loss_d.backward()
            # for generator
            with autocast(enabled=params.use_fp16):
                (
                    commit_loss,
                    gen_stft_adv_loss,
                    gen_period_adv_loss,
                    gen_scale_adv_loss,
                    feature_stft_loss,
                    feature_period_loss,
                    feature_scale_loss,
                    wav_reconstruction_loss,
                    mel_reconstruction_loss,
                    stats_g,
                ) = model(
                    speech=audio,
                    speech_lengths=audio_lens,
                    forward_generator=True,
                    return_sample=False,
                )
            loss_g = (
                (gen_stft_adv_loss + gen_period_adv_loss + gen_scale_adv_loss)
                * train_discriminator(
                    params.lambda_adv,
                    0,
                    threshold=params.discriminator_epoch_start,
                )
                + (
                    params.lambda_wav * wav_reconstruction_loss
                    + params.lambda_rec * mel_reconstruction_loss
                )
                + params.lambda_feat
                * (feature_stft_loss + feature_period_loss + feature_scale_loss)
                + params.lambda_com * commit_loss
            )
            optimizer_g.zero_grad()
            loss_g.backward()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    libritts = LibriTTSCodecDataModule(args)

    if params.full_libri:
        train_cuts = libritts.train_all_shuf_cuts()
    else:
        train_cuts = libritts.train_clean_100_cuts()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
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

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=True,
        )

    optimizer_g = torch.optim.AdamW(
        itertools.chain(
            encoder.parameters(),
            quantizer.parameters(),
            decoder.parameters(),
        ),
        lr=params.lr,
        betas=(0.5, 0.9),
    )
    discriminator_params = [
        multi_scale_stft_discriminator.parameters(),
    ]
    if multi_scale_discriminator is not None:
        discriminator_params.append(multi_scale_discriminator.parameters())
    if multi_period_discriminator is not None:
        discriminator_params.append(multi_period_discriminator.parameters())
    optimizer_d = torch.optim.AdamW(
        itertools.chain(*discriminator_params),
        lr=params.lr,
        betas=(0.5, 0.9),
    )

    scheduler_g = WarmupCosineLrScheduler(
        optimizer=optimizer_g,
        max_iter=params.num_epochs * 1500,
        eta_ratio=0.1,
        warmup_iter=params.discriminator_epoch_start * 1500,
        warmup_ratio=1e-4,
    )
    scheduler_d = WarmupCosineLrScheduler(
        optimizer=optimizer_d,
        max_iter=params.num_epochs * 1500,
        eta_ratio=0.1,
        warmup_iter=params.discriminator_epoch_start * 1500,
        warmup_ratio=1e-4,
    )

    if checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer_g" in checkpoints:
            logging.info("Loading optimizer_g state dict")
            optimizer_g.load_state_dict(checkpoints["optimizer_g"])
        if "optimizer_d" in checkpoints:
            logging.info("Loading optimizer_d state dict")
            optimizer_d.load_state_dict(checkpoints["optimizer_d"])

        # load state_dict for schedulers
        if "scheduler_g" in checkpoints:
            logging.info("Loading scheduler_g state dict")
            scheduler_g.load_state_dict(checkpoints["scheduler_g"])
        if "scheduler_d" in checkpoints:
            logging.info("Loading scheduler_d state dict")
            scheduler_d.load_state_dict(checkpoints["scheduler_d"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    train_dl = libritts.train_dataloaders(
        train_cuts,
        world_size=world_size,
        rank=rank,
    )

    valid_cuts = libritts.dev_clean_cuts()
    valid_dl = libritts.valid_dataloaders(
        valid_cuts,
        world_size=world_size,
        rank=rank,
    )

    if not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        if epoch % params.save_every_n == 0 or epoch == params.num_epochs:
            filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
            save_checkpoint(
                filename=filename,
                params=params,
                model=model,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            if rank == 0:
                if params.best_train_epoch == params.cur_epoch:
                    best_train_filename = params.exp_dir / "best-train-loss.pt"
                    copyfile(src=filename, dst=best_train_filename)

                if params.best_valid_epoch == params.cur_epoch:
                    best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                    copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    LibriTTSCodecDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
