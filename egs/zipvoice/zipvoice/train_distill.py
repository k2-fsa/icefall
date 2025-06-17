#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
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
This script trains a ZipVoice-Distill model starting from a ZipVoice model.
It has two distillation stages.

Usage:

(1) The first distillation stage with a fixed ZipVoice model as the teacher.
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "emilia" \
        --base-lr 0.0005 \
        --max-duration 500 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
        --num-updates 60000 \
        --distill-stage "first" \
        --exp-dir zipvoice/exp_zipvoice_distill_1stage

(2) The second distillation stage with a EMA model as the teacher.
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "emilia" \
        --base-lr 0.0001 \
        --max-duration 500 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_distill_1stage/iter-60000-avg-7.pt \
        --num-updates 2000 \
        --distill-stage "second" \
        --exp-dir zipvoice/exp_zipvoice_distill
"""

import argparse
import copy
import logging
import os
import random
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from checkpoint import load_checkpoint, save_checkpoint
from lhotse.cut import Cut, CutSet
from lhotse.utils import fix_random_seed
from model import get_distill_model, get_model
from optim import FixedLRScheduler, ScaledAdam
from tokenizer import TokenizerEmilia, TokenizerLibriTTS
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from train_flow import add_model_arguments, get_params
from tts_datamodule import TtsDataModule
from utils import (
    condition_time_mask,
    get_adjusted_batch_count,
    prepare_input,
    set_batch_count,
)

from icefall import diagnostics
from icefall.checkpoint import (
    remove_checkpoints,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    make_pad_mask,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


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
        default=1,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--num-updates",
        type=int,
        default=0,
        help="Number of updates to train, will ignore num_epochs if > 0.",
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
        "--teacher-model",
        type=str,
        help="""Checkpoints of pre-trained teacher model""",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipvoice/exp_zipvoice_distill",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.001, help="The base learning rate."
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=50,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
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
        "--average-period",
        type=int,
        default=200,
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
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="The EMA decay factor of target model in distillation.",
    )
    parser.add_argument(
        "--distill-stage",
        type=str,
        choices=["first", "second"],
        help="The stage of distillation.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="emilia",
        choices=["emilia", "libritts"],
        help="The used training dataset",
    )

    add_model_arguments(parser)

    return parser


def ema(new_model, ema_model, decay):
    if isinstance(new_model, DDP):
        new_model = new_model.module
    if isinstance(ema_model, DDP):
        ema_model = ema_model.module
    new_model_dict = new_model.state_dict()
    ema_model_dict = ema_model.state_dict()
    for key in new_model_dict.keys():
        ema_model_dict[key].data.copy_(
            ema_model_dict[key].data * decay + new_model_dict[key].data * (1 - decay)
        )


def resume_checkpoint(
    params: AttributeDict, model: nn.Module, model_avg: nn.Module, model_ema: nn.Module
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
    filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename, model=model, model_avg=model_avg, model_ema=model_ema, strict=True
    )

    if params.start_epoch > 1:
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


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    teacher_model: Union[nn.Module, DDP],
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      teacher_model:
        The teacher model for distillation.
      features:
        The target acoustic feature.
      features_lens:
        The number of frames of each utterance.
      tokens:
        Input tokens that representing the transcripts.
      durations:
        Duration of each token.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape

    features = torch.nn.functional.pad(
        features, (0, 0, 0, num_frames - features.size(1))
    )  # (B, T, F)
    noise = torch.randn_like(features)  # (B, T, F)

    # Sampling t and guidance_scale from uniform distribution

    t_value = random.random()
    t = torch.ones(batch_size, 1, 1, device=device) * t_value
    if params.distill_stage == "first":
        guidance_scale = torch.rand(batch_size, 1, 1, device=device) * 2
    else:
        guidance_scale = torch.rand(batch_size, 1, 1, device=device) * 2 + 1
    xt = features * t + noise * (1 - t)
    t_delta_fix = random.uniform(0.0, min(0.3, 1 - t_value))
    t_delta_ema = random.uniform(0.0, min(0.3, 1 - t_value - t_delta_fix))
    t_dest = t + t_delta_fix + t_delta_ema

    with torch.no_grad():
        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )

        if params.distill_stage == "first":
            teacher_x_t_mid, _ = teacher_model.sample_intermediate(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=xt,
                speech_condition_mask=speech_condition_mask,
                t_start=t,
                t_end=t + t_delta_fix,
                num_step=1,
                guidance_scale=guidance_scale,
            )

            target_x1, _ = teacher_model.sample_intermediate(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=teacher_x_t_mid,
                speech_condition_mask=speech_condition_mask,
                t_start=t + t_delta_fix,
                t_end=t_dest,
                num_step=1,
                guidance_scale=guidance_scale,
            )
        else:
            teacher_x_t_mid, _ = teacher_model(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=xt,
                speech_condition_mask=speech_condition_mask,
                t_start=t,
                t_end=t + t_delta_fix,
                num_step=1,
                guidance_scale=guidance_scale,
            )

            target_x1, _ = teacher_model(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=teacher_x_t_mid,
                speech_condition_mask=speech_condition_mask,
                t_start=t + t_delta_fix,
                t_end=t_dest,
                num_step=1,
                guidance_scale=guidance_scale,
            )

    with torch.set_grad_enabled(is_training):

        pred_x1, _ = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=xt,
            speech_condition_mask=speech_condition_mask,
            t_start=t,
            t_end=t_dest,
            num_step=1,
            guidance_scale=guidance_scale,
        )
        pred_v = (pred_x1 - xt) / (t_dest - t)

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)
        loss_mask = speech_condition_mask & (~padding_mask)

        target_v = (target_x1 - xt) / (t_dest - t)
        loss = torch.mean((pred_v[loss_mask] - target_v[loss_mask]) ** 2)

        ut = features - noise  # (B, T, F)

        ref_loss = torch.mean((pred_v[loss_mask] - ut[loss_mask]) ** 2)

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames
    info["ref_loss"] = ref_loss.detach().cpu().item() * num_frames
    return loss, info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    teacher_model: Union[nn.Module, DDP],
    tokenizer: TokenizerEmilia,
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
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
      teacher_model:
        The model for distillation.
      tokenizer:
        Used to convert text to tokens.
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
            model_ema=teacher_model,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params) + 100000)

        if (
            params.valid_interval is None
            and batch_idx == 0
            and not params.print_diagnostics
        ) or (
            params.valid_interval is not None
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                valid_dl=valid_dl,
                world_size=world_size,
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

        params.batch_idx_train += 1

        batch_size = len(batch["text"])

        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            tokenizer=tokenizer,
            return_tokens=True,
            return_feature=True,
        )

        try:
            with autocast("cuda", enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    teacher_model=teacher_model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if params.distill_stage == "second":
                ema(model, teacher_model, params.ema_decay)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.info(f"out of memory error at rank {rank}")
                # optimizer.zero_grad()
                # duration_optimizer.zero_grad()
                torch.cuda.empty_cache()
                raise
                continue
            else:
                logging.info(f"Caught exception : {e}.")
                save_bad_model()
                raise
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
        if (
            params.batch_idx_train > 0
            and params.num_updates > 0
            and params.batch_idx_train > params.num_updates
        ):
            break
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 1024.0 or (
                cur_grad_scale < 4096.0 and params.batch_idx_train % 400 == 0
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
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    teacher_model: Optional[nn.Module],
    tokenizer: TokenizerEmilia,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            tokenizer=tokenizer,
            return_tokens=True,
            return_feature=True,
        )

        loss, loss_info = compute_fbank_loss(
            params=params,
            model=model,
            teacher_model=teacher_model,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
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

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    os.makedirs(f"{params.exp_dir}/fbank", exist_ok=True)

    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    if params.dataset == "emilia":
        tokenizer = TokenizerEmilia(
            token_file=params.token_file, token_type=params.token_type
        )
    elif params.dataset == "libritts":
        tokenizer = TokenizerLibriTTS(
            token_file=params.token_file, token_type=params.token_type
        )

    params.vocab_size = tokenizer.vocab_size
    params.pad_id = tokenizer.pad_id

    params.device = device

    logging.info(params)

    logging.info("About to create model")

    assert params.teacher_model is not None
    logging.info(f"Loading pre-trained model from {params.teacher_model}")
    model = get_distill_model(params)
    _ = load_checkpoint(
        filename=params.teacher_model,
        model=model,
        strict=(params.distill_stage == "second"),
    )

    if params.distill_stage == "first":
        teacher_model = get_model(params)
        _ = load_checkpoint(
            filename=params.teacher_model, model=teacher_model, strict=True
        )
    else:
        teacher_model = copy.deepcopy(model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)
    assert params.start_epoch > 0, params.start_epoch
    if params.start_epoch > 1:
        logging.info(f"Resuming from epoch {params.start_epoch}")
        if params.distill_stage == "first":
            checkpoints = resume_checkpoint(
                params=params, model=model, model_avg=model_avg
            )
        else:
            checkpoints = resume_checkpoint(
                params=params, model=model, model_avg=model_avg, model_ema=teacher_model
            )

    model = model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # only update the fm_decoder
    num_trainable = 0
    for name, p in model.named_parameters():
        if "fm_decoder" in name:
            p.requires_grad = True
            num_trainable += p.numel()
        else:
            p.requires_grad = False

    logging.info(
        "A total of {} trainable parameters ({:.3f}% of the whole model)".format(
            num_trainable, num_trainable / num_param * 100
        )
    )

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = FixedLRScheduler(optimizer)

    scaler = GradScaler("cuda", enabled=params.use_fp16)

    if params.start_epoch > 1 and checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

        if "grad_scaler" in checkpoints:
            logging.info("Loading grad scaler state dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    def remove_short_and_long_utt_emilia(c: Cut):
        if c.duration < 1.0 or c.duration > 30.0:
            return False
        return True

    def remove_short_and_long_utt_libritts(c: Cut):
        if c.duration < 1.0 or c.duration > 20.0:
            return False
        return True

    datamodule = TtsDataModule(args)
    if params.dataset == "emilia":
        train_cuts = CutSet.mux(
            datamodule.train_emilia_EN_cuts(),
            datamodule.train_emilia_ZH_cuts(),
            weights=[46000, 49000],
        )
        train_cuts = train_cuts.filter(remove_short_and_long_utt_emilia)
        dev_cuts = CutSet.mux(
            datamodule.dev_emilia_EN_cuts(),
            datamodule.dev_emilia_ZH_cuts(),
            weights=[0.5, 0.5],
        )
    elif params.dataset == "libritts":
        train_cuts = datamodule.train_libritts_cuts()
        train_cuts = train_cuts.filter(remove_short_and_long_utt_libritts)
        dev_cuts = datamodule.dev_libritts_cuts()

    train_dl = datamodule.train_dataloaders(train_cuts)

    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
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
            model_ema=teacher_model,
            optimizer=optimizer,
            scheduler=scheduler,
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
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
