#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo)
# Copyright    2023                           (authors: Feiteng Li)
# Copyright    2024                           (authors: Yuekai Zhang)
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
Usage:
# docker: ghcr.io/swivid/f5-tts:main
# pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install kaldialign lhotse tensorboard bigvganinference sentencepiece

world_size=8
exp_dir=exp/f5-tts-small
python3 f5-tts/train.py --max-duration 700 --filter-min-duration 0.5 --filter-max-duration 20  \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 10000 \
      --base-lr 7.5e-5 --warmup-steps 20000 --num-epochs 60  \
      --num-decoder-layers 18 --nhead 12 --decoder-dim 768 \
      --exp-dir ${exp_dir} --world-size ${world_size}

# command for training with cosyvoice semantic token
exp_dir=exp/f5-tts-cosyvoice
python3 f5-tts/train.py --max-duration 700 --filter-min-duration 0.5 --filter-max-duration 20  \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 10000 \
      --base-lr 1e-4 --warmup-steps 20000 --average-period 0 \
      --num-epochs 10 --start-epoch 1 --start-batch 0 \
      --num-decoder-layers 18 --nhead 12 --decoder-dim 768 \
      --exp-dir ${exp_dir} --world-size ${world_size} \
      --decay-steps 600000 --prefix wenetspeech4tts_cosy_token --use-cosyvoice-semantic-token True
"""

import argparse
import copy
import logging
import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model.cfm import CFM
from model.dit import DiT
from model.utils import convert_char_to_pinyin
from torch import Tensor
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import TtsDataModule
from utils import MetricsTracker

from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, setup_logger, str2bool  # MetricsTracker

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module

    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads in the Decoder layers.",
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=22,
        help="Number of Decoder layers.",
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
        default=20,
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
        type=Path,
        default="exp/f5",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="f5-tts/vocab.txt",
        help="Path to the unique text tokens file",
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Path to file",
    )

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="AdamW",
        help="The optimizer.",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--decay-steps",
        type=int,
        default=1000000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
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
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train %% save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=10000,
        help="""Run validation if batch_idx %% valid_interval is 0.""",
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=0,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--accumulate-grad-steps",
        type=int,
        default=1,
        help="""update gradient when batch_idx_train %% accumulate_grad_steps == 0.
        """,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Training dtype: float32 bfloat16 float16.",
    )

    parser.add_argument(
        "--filter-min-duration",
        type=float,
        default=0.0,
        help="Keep only utterances with duration > this.",
    )

    parser.add_argument(
        "--filter-max-duration",
        type=float,
        default=20.0,
        help="Keep only utterances with duration < this.",
    )

    parser.add_argument(
        "--oom-check",
        type=str2bool,
        default=False,
        help="perform OOM check on dataloader batches before starting training.",
    )

    parser.add_argument(
        "--use-cosyvoice-semantic-token",
        type=str2bool,
        default=False,
        help="Whether to use cosyvoice semantic token to replace text token.",
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
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,
            "reset_interval": 200,
            "valid_interval": 10000,
            "env_info": get_env_info(),
        }
    )

    return params


def get_tokenizer(vocab_file_path: str):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


def get_model(params):
    if params.use_cosyvoice_semantic_token:
        # https://www.modelscope.cn/models/iic/CosyVoice2-0.5B/file/view/master?fileName=cosyvoice.yaml&status=1#L36
        vocab_char_map, vocab_size = None, 6561
    else:
        vocab_char_map, vocab_size = get_tokenizer(params.tokens)
    # bigvgan 100 dim features
    n_mel_channels = 100
    n_fft = 1024
    sampling_rate = 24_000
    hop_length = 256
    win_length = 1024

    model_cfg = {
        "dim": params.decoder_dim,
        "depth": params.num_decoder_layers,
        "heads": params.nhead,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
        "checkpoint_activations": False,
    }
    model = CFM(
        transformer=DiT(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=sampling_rate,
            mel_spec_type="bigvgan",
        ),
        odeint_kwargs=dict(
            method="euler",
        ),
        vocab_char_map=vocab_char_map,
    )
    return model


def load_F5_TTS_pretrained_checkpoint(
    model, ckpt_path, device: str = "cpu", dtype=torch.float32
):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "ema_model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

    # patch for backward compatibility, 305e3ea
    for key in [
        "mel_spec.mel_stft.mel_scale.fb",
        "mel_spec.mel_stft.spectrogram.window",
    ]:
        if key in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"][key]
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
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

    if isinstance(model, DDP):
        raise ValueError("load_checkpoint before DDP")

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
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


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def interpolate_tokens(cosy_tokens, pad_token=-1):
    """Interpolate cosyvoice tokens to match bigvgan frames length"""
    # cosyvoice, 25 tokens/sec
    # bigvgan    sample_rate/hop_length   24000/256 frames/sec
    # For every 4 cosyvoice tokens, insert pad tokens to extend it to 15 tokens to match bigvgan frames length
    # We choose 4,4,4,3 to match 15 frames
    three, two = [pad_token] * 3, [pad_token] * 2
    return [
        x
        for i, e in enumerate(cosy_tokens)
        for x in ([e] + three if i % 4 < 3 else [e] + two)
    ]


def prepare_input(
    batch: dict, device: torch.device, use_cosyvoice_semantic_token: bool
):
    """Parse batch data"""
    mel_spec = batch["features"]
    mel_lengths = batch["features_lens"]

    if use_cosyvoice_semantic_token:
        semantic_tokens = []
        for i in range(len(batch["tokens"])):
            tokens = batch["tokens"][i]
            tokens = interpolate_tokens(tokens)
            semantic_tokens.append(tokens)
        # pad to the same length, B,T, with pad value -1
        max_len = max([len(tokens) for tokens in semantic_tokens])
        text_inputs = torch.full(
            (len(semantic_tokens), max_len), -1, dtype=torch.long
        ).to(device)
        for i, tokens in enumerate(semantic_tokens):
            text_inputs[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    else:
        text_inputs = batch["text"]
        text_inputs = convert_char_to_pinyin(text_inputs, polyphone=True)

    return text_inputs, mel_spec.to(device), mel_lengths.to(device)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    (text_inputs, mel_spec, mel_lengths) = prepare_input(
        batch,
        device=device,
        use_cosyvoice_semantic_token=params.use_cosyvoice_semantic_token,
    )
    # at entry, TextTokens is (N, P)

    with torch.set_grad_enabled(is_training):
        loss, cond, pred = model(mel_spec, text=text_inputs, lens=mel_lengths)
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["samples"] = mel_lengths.size(0)

    info["loss"] = loss.detach().cpu().item() * info["samples"]

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
    if world_size > 1:
        tot_loss.reduce(loss.device)
    loss_value = tot_loss["loss"] / tot_loss["samples"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    rng: random.Random,
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
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      rng:
        Random for selecting.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    tot_loss = MetricsTracker()
    iter_dl = iter(train_dl)

    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True

    batch_idx = 0
    while True:
        try:
            batch = next(iter_dl)
        except StopIteration:
            logging.info("Reaches end of dataloader.")
            break

        batch_idx += 1

        params.batch_idx_train += 1
        batch_size = len(batch["text"])

        try:
            with torch.amp.autocast("cuda", dtype=dtype, enabled=enabled):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    is_training=True,
                )

            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info * (
                1 / params.reset_interval
            )

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.

            scaler.scale(loss).backward()
            if params.batch_idx_train >= params.accumulate_grad_steps:
                if params.batch_idx_train % params.accumulate_grad_steps == 0:

                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()

                    for k in range(params.accumulate_grad_steps):
                        scheduler.step()

            set_batch_count(model, params.batch_idx_train)
        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if params.average_period > 0:
            if (
                params.batch_idx_train > 0
                and params.batch_idx_train % params.average_period == 0
            ):
                # Perform Operation in rank 0
                if rank == 0:
                    update_averaged_model(
                        params=params,
                        model_cur=model,
                        model_avg=model_avg,
                    )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            # Perform Operation in rank 0
            if rank == 0:
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

        if batch_idx % 100 == 0 and params.dtype in ["float16", "fp16"]:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)

            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = (
                scaler._scale.item() if params.dtype in ["float16", "fp16"] else 1.0
            )

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, train_loss[{loss_info}], "
                f"batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
                + (
                    f", grad_scale: {cur_grad_scale}"
                    if params.dtype in ["float16", "fp16"]
                    else ""
                )
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer,
                    "train/current_",
                    params.batch_idx_train,
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.dtype in ["float16", "fp16"]:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

        if params.batch_idx_train % params.valid_interval == 0:
            # Calculate validation loss in Rank 0
            model.eval()
            logging.info("Computing validation loss")
            with torch.amp.autocast("cuda", dtype=dtype):
                valid_info = compute_validation_loss(
                    params=params,
                    model=model,
                    tokenizer=tokenizer,
                    valid_dl=valid_dl,
                    world_size=world_size,
                )
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )

            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

            model.train()

    loss_value = tot_loss["loss"] / tot_loss["samples"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def filter_short_and_long_utterances(
    cuts: CutSet, min_duration: float, max_duration: float
) -> CutSet:
    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 0.6 second and 20 seconds
        if c.duration < min_duration or c.duration > max_duration:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    cuts = cuts.filter(remove_short_and_long_utt)

    return cuts


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
    rng = random.Random(params.seed)
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
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"Device: {device}")
    tokenizer = get_tokenizer(params.tokens)
    logging.info(params)

    logging.info("About to create model")

    model = get_model(params)

    if params.pretrained_model_path:
        checkpoint = torch.load(params.pretrained_model_path, map_location="cpu", weights_only=False)
        if "ema_model_state_dict" in checkpoint or "model_state_dict" in checkpoint:
            model = load_F5_TTS_pretrained_checkpoint(
                model, params.pretrained_model_path
            )
        else:
            _ = load_checkpoint(
                params.pretrained_model_path,
                model=model,
            )

    model = model.to(device)

    with open(f"{params.exp_dir}/model.txt", "w") as f:
        print(model)
        print(model, file=f)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch

    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    model_parameters = model.parameters()

    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=params.base_lr,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
        eps=1e-8,
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=params.warmup_steps
    )
    decay_scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=1e-8, total_iters=params.decay_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[params.warmup_steps],
    )

    optimizer.zero_grad()

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.inf_check:
        register_inf_check_hooks(model)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    dataset = TtsDataModule(args)
    train_cuts = dataset.train_cuts()
    valid_cuts = dataset.valid_cuts()

    train_cuts = filter_short_and_long_utterances(
        train_cuts, params.filter_min_duration, params.filter_max_duration
    )
    valid_cuts = filter_short_and_long_utterances(
        valid_cuts, params.filter_min_duration, params.filter_max_duration
    )

    train_dl = dataset.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    valid_dl = dataset.valid_dataloaders(valid_cuts)

    if params.oom_check:
        scan_pessimistic_batches_for_oom(
            model=model,
            tokenizer=tokenizer,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    scaler = GradScaler(
        "cuda", enabled=(params.dtype in ["fp16", "float16"]), init_scale=1.0
    )
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            tokenizer=tokenizer,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            rng=rng,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    tokenizer,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    dtype = torch.float32
    if params.dtype in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif params.dtype in ["float16", "fp16"]:
        dtype = torch.float16

    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        print(batch.keys())
        try:
            with torch.amp.autocast("cuda", dtype=dtype):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    is_training=True,
                )
            loss.backward(retain_graph=True)
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()

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
