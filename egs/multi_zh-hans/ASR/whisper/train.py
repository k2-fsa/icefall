#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
#              2024  Yuekai Zhang
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

#fine-tuning with deepspeed zero stage 1
torchrun --nproc-per-node 8 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json

# fine-tuning with ddp
torchrun --nproc_per_node 8 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir whisper/exp_medium \
  --base-lr 1e-5 \
  --model-name medium
"""

import argparse
import copy
import logging
import os
import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import k2
import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import whisper
from asr_datamodule import AsrDataModule
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from label_smoothing import LabelSmoothingLoss
from lhotse import CutSet, load_manifest
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from multi_dataset import MultiDataset
from optim import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.functional import pad as pad_tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from whisper_decoder_forward_monkey_patch import replace_whisper_decoder_forward
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import update_averaged_model
from icefall.dist import cleanup_dist, get_rank, get_world_size, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    filter_uneven_sized_batch,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default=10,
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
        default="whisper/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="large-v2",
        choices=["large-v2", "large-v3", "medium", "base", "small", "tiny"],
        help="""The model name to use.
        """,
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="""The path to the pretrained model if it is not None. Training will
        start from this model. e.g. ./wenetspeech/ASR/whisper/exp_large_v2/epoch-4-avg-3.pt
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=1e-5, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
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
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-distill-whisper",
        type=str2bool,
        default=False,
        help="Whether to use architecture of distill whisper.",
    )

    parser = deepspeed.add_config_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - frame_shift_ms: The frame shift in milliseconds.
        - allowed_excess_duration_ratio: The allowed excess duration ratio.
        - best_train_loss: The best training loss so far.
        - best_valid_loss: The best validation loss so far.
        - best_train_epoch: The epoch where the best training loss is achieved.
        - best_valid_epoch: The epoch where the best validation loss is achieved.
        - batch_idx_train: The batch index of the current batch.
        - log_interval: Log training stats every `log_interval` batches.
        - reset_interval: Reset the stats every `reset_interval` batches.
        - valid_interval: Run validation every `valid_interval` batches.
        - env_info: The environment information.
    """
    params = AttributeDict(
        {
            "frame_shift_ms": 10.0,
            "subsampling_factor": 2,
            "allowed_excess_duration_ratio": 0.1,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 10000,
            "env_info": get_env_info(),
        }
    )

    return params


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


def compute_loss(
    params: AttributeDict,
    tokenizer: whisper.tokenizer.Tokenizer,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute the loss for the given batch.
    Args:
        params:
            It is returned by :func:`get_params`.
        tokenizer:
            The tokenizer used to encode the text.
        model:
            The model for training.
        batch:
            A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
            for the content in it.
        is_training:
            Whether it is training.
    Returns:
        Return a tuple of two elements. The first element is the loss tensor.
    """
    # For the uneven-sized batch, the total duration after padding would possibly
    # cause OOM. Hence, for each batch, which is sorted descendingly by length,
    # we simply drop the last few shortest samples, so that the retained total frames
    # (after padding) would not exceed `allowed_max_frames`:
    # `allowed_max_frames = int(max_frames * (1.0 + allowed_excess_duration_ratio))`,
    # where `max_frames = max_duration * 1000 // frame_shift_ms`.
    # We set allowed_excess_duration_ratio=0.1.
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module

    def _batch_tensors(tensors: List[Tensor], pad_value: Any) -> Tensor:
        padding_size = max(tensor.shape[0] for tensor in tensors)
        dims = len(tensors[0].shape)
        padded_tensors = []
        for tensor in tensors:
            padding = [0] * 2 * dims
            padding[-1] = padding_size - tensor.shape[0]
            padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
        return torch.stack([tensor for tensor in padded_tensors], dim=0)

    def normalize_text_alimeeting(text: str, normalize: str = "m2met") -> str:
        """
        Text normalization similar to M2MeT challenge baseline.
        See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
        """
        if normalize == "none":
            return text
        elif normalize == "m2met":
            import re

            text = text.replace(" ", "")
            text = text.replace("<sil>", "")
            text = text.replace("<%>", "")
            text = text.replace("<->", "")
            text = text.replace("<$>", "")
            text = text.replace("<#>", "")
            text = text.replace("<_>", "")
            text = text.replace("<space>", "")
            text = text.replace("`", "")
            text = text.replace("&", "")
            text = text.replace(",", "")
            if re.search("[a-zA-Z]", text):
                text = text.upper()
            text = text.replace("Ａ", "A")
            text = text.replace("ａ", "A")
            text = text.replace("ｂ", "B")
            text = text.replace("ｃ", "C")
            text = text.replace("ｋ", "K")
            text = text.replace("ｔ", "T")
            text = text.replace("，", "")
            text = text.replace("丶", "")
            text = text.replace("。", "")
            text = text.replace("、", "")
            text = text.replace("？", "")
            return text

    max_frames = params.max_duration * 1000 // params.frame_shift_ms
    allowed_max_frames = int(max_frames * (1.0 + params.allowed_excess_duration_ratio))
    batch = filter_uneven_sized_batch(batch, allowed_max_frames)

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]

    assert feature.ndim == 3
    feature = feature.to(device)
    feature = feature.transpose(1, 2)  # (N, C, T)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train

    texts = batch["supervisions"]["text"]
    # remove spaces in texts
    texts = [normalize_text_alimeeting(text) for text in texts]

    text_tokens_list = [
        list(tokenizer.sot_sequence_including_notimestamps)
        + tokenizer.encode(text)
        + [tokenizer.eot]
        for text in texts
    ]
    # convert it to torch tensor
    text_tokens_list = [
        torch.LongTensor(text_tokens) for text_tokens in text_tokens_list
    ]

    # 50256 is the index of <pad> for all whisper models
    prev_outputs_tokens = _batch_tensors(
        [tokens[:-1] for tokens in text_tokens_list], pad_value=50256
    )
    target_tokens = _batch_tensors(
        [tokens[1:] for tokens in text_tokens_list], pad_value=50256
    )
    target_lengths = torch.LongTensor(
        [tokens.shape[0] - 1 for tokens in text_tokens_list]
    )

    decoder_criterion = LabelSmoothingLoss(
        ignore_index=50256, label_smoothing=0.1, reduction="sum"
    )

    # ignore the first 3 tokens, which are always <|lang_id|>, <|transcibe|>, <|notimestampes|>
    ignore_prefix_size = 3
    with torch.set_grad_enabled(is_training):
        encoder_out = model.encoder(feature)
        text_logits = model.decoder(prev_outputs_tokens.to(device), encoder_out)
        text_logits = text_logits[:, ignore_prefix_size:, :]
        target_tokens = target_tokens[:, ignore_prefix_size:]
        loss = decoder_criterion(text_logits, target_tokens.to(device))

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    tokenizer: whisper.tokenizer.Tokenizer,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                batch=batch,
                is_training=False,
            )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    tokenizer: whisper.tokenizer.Tokenizer,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
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
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
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

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
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
            if params.deepspeed:
                model.save_checkpoint(
                    save_dir=params.exp_dir,
                    tag=f"epoch-{params.cur_epoch}-checkpoint-{batch_idx}",
                    client_state={},
                )
                if rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(
                        params.exp_dir,
                        f"{params.exp_dir}/epoch-{params.cur_epoch}-checkpoint-{batch_idx}.pt",
                        tag=f"epoch-{params.cur_epoch}-checkpoint-{batch_idx}",
                    )
                    os.system(
                        f"rm -rf {params.exp_dir}/epoch-{params.cur_epoch}-checkpoint-{batch_idx}"
                    )

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    tokenizer=tokenizer,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            if params.deepspeed:
                # deepspeed's backward() is different from torch's backward()
                # in that it does not accept a loss tensor as input.
                # It computes the loss internally.
                model.backward(loss)
                model.step()
            else:
                scaler.scale(loss).backward()
                set_batch_count(model, params.batch_idx_train)
                scheduler.step_batch(params.batch_idx_train)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
            and not params.deepspeed
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if batch_idx % 100 == 0 and params.use_fp16 and not params.deepspeed:
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
            try:
                cur_lr = scheduler.get_last_lr()[0]
            except:  # noqa
                cur_lr = 0.0
            cur_grad_scale = (
                scaler._scale.item()
                if (params.use_fp16 and not params.deepspeed)
                else 1.0
            )

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (
                    f"grad_scale: {scaler._scale.item()}"
                    if (params.use_fp16 and not params.deepspeed)
                    else ""
                )
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
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


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

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info(params)

    logging.info("About to create model")

    replace_whisper_encoder_forward()
    if params.use_distill_whisper:
        replace_whisper_decoder_forward()
    model = whisper.load_model(params.model_name, "cpu")
    del model.alignment_heads

    if params.pretrained_model_path:
        checkpoint = torch.load(params.pretrained_model_path, map_location="cpu")
        if "model" not in checkpoint:
            model.load_state_dict(checkpoint, strict=True)
        else:
            load_checkpoint(params.pretrained_model_path, model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="zh",
        task="transcribe",
    )

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.base_lr)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

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

    if world_size > 1:
        if params.deepspeed:
            logging.info("Using DeepSpeed")
            model, optimizer, _, scheduler = deepspeed.initialize(
                args=params, model=model, model_parameters=model.parameters()
            )
        else:
            logging.info("Using DDP")
            setup_dist(use_ddp_launch=True)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    data_module = AsrDataModule(args)
    multi_dataset = MultiDataset(args.manifest_dir)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 20.0:
            # logging.warning(
            #    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    train_cuts = multi_dataset.train_cuts()
    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    train_dl = data_module.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = multi_dataset.dev_cuts()
    valid_dl = data_module.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    logging.info(f"start training from epoch {params.start_epoch}")
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if not params.deepspeed:
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            tokenizer=tokenizer,
            model=model,
            model_avg=model_avg,
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

        if params.deepspeed:
            model.save_checkpoint(
                save_dir=params.exp_dir,
                tag=f"epoch-{params.cur_epoch}",
                client_state={},
            )
            if rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(
                    params.exp_dir,
                    f"{params.exp_dir}/epoch-{params.cur_epoch}.pt",
                    tag=f"epoch-{params.cur_epoch}",
                )
                os.system(f"rm -rf {params.exp_dir}/epoch-{params.cur_epoch}")
        else:
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

    if world_size > 1 and not params.deepspeed:
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

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = get_world_size()
    rank = get_rank()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(rank=rank, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
