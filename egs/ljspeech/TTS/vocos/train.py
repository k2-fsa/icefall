#!/usr/bin/env python3
# Copyright         2023-2024  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                               Wei Kang)
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
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools
import json
import copy
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import LJSpeechTtsDataModule

from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.optim import Optimizer

from utils import (
    load_checkpoint,
    save_checkpoint,
    plot_spectrogram,
    get_cosine_schedule_with_warmup,
    save_checkpoint_with_global_batch_idx,
)

from icefall import diagnostics
from icefall.checkpoint import remove_checkpoints, update_averaged_model
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    setup_logger,
    str2bool,
    get_parameter_groups_with_lrs,
)
from model import Vocos
from lhotse import Fbank, FbankConfig


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of ConvNeXt layers.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dim of ConvNeXt module.",
    )

    parser.add_argument(
        "--intermediate-dim",
        type=int,
        default=1536,
        help="Intermediate dim of ConvNeXt module.",
    )

    parser.add_argument(
        "--max-seconds",
        type=int,
        default=60,
        help="""
        The length of the precomputed normalization window sum square
        (required by istft). This argument is only for onnx export, it determines
        the max length of the audio that be properly normalized.
        Note, you can generate audios longer than this value with the exported onnx model,
        the part longer than this value will not be normalized yet.
        The larger this value is the bigger the exported onnx model will be.
        """,
    )


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
        "--num-epochs",
        type=int,
        default=100,
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
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="vocos/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--learning-rate", type=float, default=0.0005, help="The learning rate."
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
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--keep-last-epoch-k",
        type=int,
        default=50,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=500,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--mrd-loss-scale",
        type=float,
        default=0.1,
        help="The scale of MultiResolutionDiscriminator loss.",
    )

    parser.add_argument(
        "--mel-loss-scale",
        type=float,
        default=45,
        help="The scale of melspectrogram loss.",
    )

    add_model_arguments(parser)

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

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 500,
            "feature_dim": 80,
            "segment_size": 16384,
            "adam_b1": 0.9,
            "adam_b2": 0.99,
            "warmup_steps": 0,
            "max_steps": 2000000,
            "env_info": get_env_info(),
        }
    )

    return params


def get_model(params: AttributeDict) -> nn.Module:
    device = params.device
    model = Vocos(
        feature_dim=params.feature_dim,
        dim=params.hidden_dim,
        n_fft=params.frame_length,
        hop_length=params.frame_shift,
        intermediate_dim=params.intermediate_dim,
        num_layers=params.num_layers,
        sample_rate=params.sampling_rate,
        max_seconds=params.max_seconds,
    ).to(device)

    num_param_gen = sum([p.numel() for p in model.generator.parameters()])
    logging.info(f"Number of Generator parameters : {num_param_gen}")
    num_param_mpd = sum([p.numel() for p in model.mpd.parameters()])
    logging.info(f"Number of MultiPeriodDiscriminator parameters : {num_param_mpd}")
    num_param_mrd = sum([p.numel() for p in model.mrd.parameters()])
    logging.info(f"Number of MultiResolutionDiscriminator parameters : {num_param_mrd}")
    logging.info(
        f"Number of model parameters : {num_param_gen + num_param_mpd + num_param_mrd}"
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer_g: Optional[Optimizer] = None,
    optimizer_d: Optional[Optimizer] = None,
    scheduler_g: Optional[LRScheduler] = None,
    scheduler_d: Optional[LRScheduler] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def compute_generator_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    audios: Tensor,
) -> Tuple[Tensor, MetricsTracker]:
    device = params.device
    model = model.module if isinstance(model, DDP) else model

    audios_hat = model(features)  # (B, T)

    mel_loss = model.melspec_loss(audios_hat, audios)

    _, gen_score_mpd, fmap_rs_mpd, fmap_gs_mpd = model.mpd(y=audios, y_hat=audios_hat)
    _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = model.mrd(y=audios, y_hat=audios_hat)

    loss_gen_mpd, list_loss_gen_mpd = model.gen_loss(disc_outputs=gen_score_mpd)
    loss_gen_mrd, list_loss_gen_mrd = model.gen_loss(disc_outputs=gen_score_mrd)

    loss_gen_mpd = loss_gen_mpd / len(list_loss_gen_mpd)
    loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)

    loss_fm_mpd = model.feat_matching_loss(
        fmap_r=fmap_rs_mpd, fmap_g=fmap_gs_mpd
    ) / len(fmap_rs_mpd)
    loss_fm_mrd = model.feat_matching_loss(
        fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd
    ) / len(fmap_rs_mrd)

    loss_gen_all = (
        loss_gen_mpd
        + params.mrd_loss_scale * loss_gen_mrd
        + loss_fm_mpd
        + params.mrd_loss_scale * loss_fm_mrd
        + params.mel_loss_scale * mel_loss
    )

    assert loss_gen_all.requires_grad == True

    info = MetricsTracker()
    info["frames"] = 1
    info["loss_gen"] = loss_gen_all.detach().cpu().item()
    info["loss_mel"] = mel_loss.detach().cpu().item()
    info["loss_feature_mpd"] = loss_fm_mpd.detach().cpu().item()
    info["loss_feature_mrd"] = loss_fm_mrd.detach().cpu().item()
    info["loss_gen_mrd"] = loss_gen_mrd.detach().cpu().item()
    info["loss_gen_mpd"] = loss_gen_mpd.detach().cpu().item()

    return loss_gen_all, info


def compute_discriminator_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    audios: Tensor,
) -> Tuple[Tensor, MetricsTracker]:
    device = params.device
    model = model.module if isinstance(model, DDP) else model

    with torch.no_grad():
        audios_hat = model(features)  # (B, 1, T)

    real_score_mpd, gen_score_mpd, _, _ = model.mpd(y=audios, y_hat=audios_hat)
    real_score_mrd, gen_score_mrd, _, _ = model.mrd(y=audios, y_hat=audios_hat)
    loss_mpd, loss_mpd_real, loss_mpd_gen = model.disc_loss(
        disc_real_outputs=real_score_mpd, disc_generated_outputs=gen_score_mpd
    )
    loss_mrd, loss_mrd_real, loss_mrd_gen = model.disc_loss(
        disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
    )
    loss_mpd /= len(loss_mpd_real)
    loss_mrd /= len(loss_mrd_real)

    loss_disc_all = loss_mpd + params.mrd_loss_scale * loss_mrd

    info = MetricsTracker()
    # MetricsTracker will norm the loss value with "frames", set it to 1 here to
    # make tot_loss look normal.
    info["frames"] = 1
    info["loss_disc"] = loss_disc_all.detach().cpu().item()
    info["loss_disc_mrd"] = loss_mrd.detach().cpu().item()
    info["loss_disc_mpd"] = loss_mpd.detach().cpu().item()

    return loss_disc_all, info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer_g: Optimizer,
    optimizer_d: Optimizer,
    scheduler_g: ExponentialLR,
    scheduler_d: ExponentialLR,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
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
        The model for training.
      optimizer:
        The optimizer.
      scheduler:
        The learning rate scheduler, we call step() every epoch.
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

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
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
        batch_size = batch["features_lens"].size(0)

        features = batch["features"].to(device)  # (B, T, F)
        features_lens = batch["features_lens"].to(device)
        audios = batch["audio"].to(device)

        segment_frames = (
            params.segment_size - params.frame_length
        ) // params.frame_shift + 1

        start_p = random.randint(0, features_lens.min() - (segment_frames + 1))

        features = features[:, start_p : start_p + segment_frames, :].permute(
            0, 2, 1
        )  # (B, F, T)

        audios = audios[
            :,
            start_p * params.frame_shift : start_p * params.frame_shift
            + params.segment_size,
        ]  # (B, T)

        try:
            optimizer_d.zero_grad()

            loss_disc, loss_disc_info = compute_discriminator_loss(
                params=params,
                model=model,
                features=features,
                audios=audios,
            )

            loss_disc.backward()
            optimizer_d.step()
            scheduler_d.step()

            optimizer_g.zero_grad()
            loss_gen, loss_gen_info = compute_generator_loss(
                params=params,
                model=model,
                features=features,
                audios=audios,
            )

            loss_gen.backward()
            optimizer_g.step()
            scheduler_g.step()

            loss_info = loss_gen_info + loss_disc_info
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_gen_info

        except Exception as e:
            logging.info(f"Caught exception : {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

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
                f"cur_lr_g: {cur_lr_g:.4e}, "
                f"cur_lr_d: {cur_lr_d:.4e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate_gen", cur_lr_g, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/learning_rate_disc", cur_lr_d, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if (
            params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                rank=rank,
                tb_writer=tb_writer,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss_gen"]
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
    tb_writer: Optional[SummaryWriter] = None,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    torch.cuda.empty_cache()
    model = model.module if isinstance(model, DDP) else model
    device = next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    with torch.no_grad():
        infer_time = 0
        audio_time = 0
        for batch_idx, batch in enumerate(valid_dl):
            features = batch["features"]  # (B, T, F)
            features_lens = batch["features_lens"]

            audio_time += torch.sum(features_lens)

            x = features.permute(0, 2, 1)  # (B, F, T)
            y = batch["audio"].to(device)  # (B, T)

            start = time.time()
            y_g_hat = model(x.to(device))  # (B, T)
            infer_time += time.time() - start

            if y_g_hat.size(1) > y.size(1):
                y = torch.cat(
                    [
                        y,
                        torch.zeros(
                            (y.size(0), y_g_hat.size(1) - y.size(1)), device=device
                        ),
                    ],
                    dim=1,
                )
            else:
                y = y[:, 0 : y_g_hat.size(1)]

            loss_mel_error = model.melspec_loss(y_g_hat, y)

            loss_info = MetricsTracker()
            # MetricsTracker will norm the loss value with "frames", set it to 1 here to
            # make tot_loss look normal.
            loss_info["frames"] = 1
            loss_info["loss_mel_error"] = loss_mel_error.item()

            tot_loss = tot_loss + loss_info

            if batch_idx <= 5 and rank == 0 and tb_writer is not None:
                if params.batch_idx_train == params.valid_interval:
                    tb_writer.add_audio(
                        "gt/y_{}".format(batch_idx),
                        y[0],
                        params.batch_idx_train,
                        params.sampling_rate,
                    )
                tb_writer.add_audio(
                    "generated/y_hat_{}".format(batch_idx),
                    y_g_hat[0],
                    params.batch_idx_train,
                    params.sampling_rate,
                )

        logging.info(f"Validation RTF : {infer_time / (audio_time * 10 / 1000)}")

        if world_size > 1:
            tot_loss.reduce(device)

        loss_value = tot_loss["loss_mel_error"]
        if loss_value < params.best_valid_loss:
            params.best_valid_epoch = params.cur_epoch
            params.best_valid_loss = loss_value

        return tot_loss


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
    torch.autograd.set_detect_anomaly(True)

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
    params.device = device
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model = model.to(device)
    generator = model.generator
    mrd = model.mrd
    mpd = model.mpd
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        params.learning_rate,
        betas=[params.adam_b1, params.adam_b2],
    )
    optimizer_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        params.learning_rate,
        betas=[params.adam_b1, params.adam_b2],
    )

    scheduler_g = get_cosine_schedule_with_warmup(
        optimizer_g,
        num_warmup_steps=params.warmup_steps,
        num_training_steps=params.max_steps,
    )
    scheduler_d = get_cosine_schedule_with_warmup(
        optimizer_d,
        num_warmup_steps=params.warmup_steps,
        num_training_steps=params.max_steps,
    )

    if checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer_g" in checkpoints:
            logging.info("Loading generator optimizer state dict")
            optimizer_g.load_state_dict(checkpoints["optimizer_g"])
        if "optimizer_d" in checkpoints:
            logging.info("Loading discriminator optimizer state dict")
            optimizer_d.load_state_dict(checkpoints["optimizer_d"])

        # load state_dict for schedulers
        if "scheduler_g" in checkpoints:
            logging.info("Loading generator scheduler state dict")
            scheduler_g.load_state_dict(checkpoints["scheduler_g"])
        if "scheduler_d" in checkpoints:
            logging.info("Loading discriminator scheduler state dict")
            scheduler_d.load_state_dict(checkpoints["scheduler_d"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    ljspeech = LJSpeechTtsDataModule(args)

    train_cuts = ljspeech.train_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 20.0:
            return False
        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    train_dl = ljspeech.train_dataloaders(train_cuts)

    valid_cuts = ljspeech.valid_cuts()
    valid_dl = ljspeech.valid_dataloaders(valid_cuts)

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
            model_avg=model_avg,
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

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        if params.best_train_epoch == params.cur_epoch:
            best_train_filename = params.exp_dir / "best-train-loss.pt"
            copyfile(src=filename, dst=best_train_filename)

        if params.best_valid_epoch == params.cur_epoch:
            best_valid_filename = params.exp_dir / "best-valid-loss.pt"
            copyfile(src=filename, dst=best_valid_filename)

        remove_checkpoints(
            out_dir=params.exp_dir,
            topk=params.keep_last_epoch_k,
            prefix="epoch",
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
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
