#!/usr/bin/env python3
# Copyright    2021-2025  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey,
#                                                       Xiaoyu Yang,
#                                                       Yifan Yang)
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
import copy
import json
import logging
import math
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from asr_datamodule import DataModule
from clap_module import ClipLoss
from lhotse.utils import fix_random_seed
from model import CLAP
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer
from zipformer2 import SimpleDownsample, Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import (
    cleanup_dist,
    get_local_rank,
    get_rank,
    get_world_size,
    setup_dist,
)
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    compare_model,
    create_grad_scaler,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def unwrap_model(model: Union[nn.Module, DDP]) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    # Note that we add a very large constant here to make the ScheduledFloat
    # variable as their end value.
    batch_count = (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )
    if params.large_batch_count:
        batch_count += 100000
    return batch_count


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    model = unwrap_model(model)
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_finetune_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--do-finetune",
        type=str2bool,
        default=True,
        help="If true, finetune from a pre-trained checkpoint",
    )

    parser.add_argument(
        "--init-modules",
        type=str,
        default=None,
        help="""
        Modules to be initialized. It matches all parameters starting with
        a specific key. The keys are given with Comma seperated. If None,
        all modules will be initialised. For example, if you only want to
        initialise all parameters staring with "encoder", use "encoder";
        if you want to initialise parameters starting with encoder or decoder,
        use "encoder,joiner".
        """,
    )

    parser.add_argument(
        "--finetune-ckpt",
        type=str,
        default=None,
        help="Fine-tuning from which checkpoint (path to a .pt file)",
    )

    parser.add_argument(
        "--freeze-encoder",
        type=str2bool,
        default=False,
        help="Freeze the encoder of the model. If true, freeze-encoder-steps won't be used",
    )

    parser.add_argument(
        "--freeze-encoder-steps",
        type=int,
        default=-1,
        help="For this number of steps, freeze the encoder. If set, freeze-encoder cannot be true; -1 means not freezing",
    )

    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
    )


def add_model_arguments(parser: argparse.ArgumentParser):
    # audio encoder
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--output-downsampling-factor",
        type=int,
        default=2,
        help="The outout downsampling factor. Default is 2. If 1, no downsample is performed.",
    )

    parser.add_argument(
        "--post-encoder-downsampling-factor",
        type=int,
        default=1,
        help="The downsampling factor after the zipformer encoder",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    # text encoder
    parser.add_argument(
        "--text-encoder-dim",
        type=int,
        default=768,
        help="Embedding dimension in the text encoder model.",
    )

    # joiner
    parser.add_argument(
        "--joint-dim",
        type=int,
        default=512,
        help="Dimension used in the joiner model.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--use-multi-node",
        type=str2bool,
        default=False,
        help="""True if using multi-node multi-GPU.
        You are not supposed to set it directly.
        """,
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
        default=30,
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
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.0045,
        help="""The base learning rate.
        It is set to a very small value as we are doing fine-tuning""",
    )

    parser.add_argument(
        "--warmup-start",
        type=float,
        default=0.5,
        help="The initial value of warmup, between 0 and 1",
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=500.0,
        help="The number of batches for lr warmup",
    )

    parser.add_argument(
        "--large-batch-count",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=100000.0,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. It is set to a very large value here to prevent the lr from decaying too fast
        during fine-tuning.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=100.0,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        It is set to a very large value here to prevent the lr from decaying too fast
        during fine-tuning.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
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
        default=100000,
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
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
    )

    parser.add_argument(
        "--use-local-loss",
        type=str2bool,
        default=False,
        help="""Whether to use local-only CLIP loss. If True, no cross-GPU
        feature gather is performed, which saves communication and memory
        but may reduce performance.
        """,
    )

    parser.add_argument(
        "--gather-with-grad",
        type=str2bool,
        default=False,
        help="""Whether to allow gradients to flow through cross-GPU feature
        gathering during Clip loss computation. If True, the gathered
        global features retain gradient and participate in back-propagation,
        providing a more complete optimization signal but increasing
        communication cost and memory usage. If False, gathered features are
        detached, reducing overhead but gradients only flow for local samples.
        """,
    )

    add_model_arguments(parser)
    add_finetune_arguments(parser)

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

        - best_train_epoch: It is the epoch that has the best training loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_train_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 128,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_criterion(params: AttributeDict, world_size: int, rank: int) -> nn.Module:
    criterion = ClipLoss(
        local_loss=params.use_local_loss,
        gather_with_grad=params.gather_with_grad,
        world_size=world_size,
        rank=rank,
    )
    return criterion


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if params.output_downsampling_factor == 2:
        assert (
            params.post_encoder_downsampling_factor == 1
        ), "CANNOT perform double output downsample!"

    encoder = Zipformer2(
        output_downsampling_factor=params.output_downsampling_factor,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_encoder_downsample_module(params: AttributeDict) -> nn.Module:
    if params.post_encoder_downsampling_factor > 1:
        downsample_module = SimpleDownsample(
            max(_to_int_tuple(params.encoder_dim)),
            downsample=params.post_encoder_downsampling_factor,
            dropout=0.0,
        )
    else:
        downsample_module = None
    return downsample_module


def get_model(params: AttributeDict) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    post_encoder_downsample = get_encoder_downsample_module(params)

    # modify the subsampling_factor accordingly
    if params.output_downsampling_factor == 1:
        params.subsampling_factor = 2

    model = CLAP(
        encoder_embed=encoder_embed,
        encoder=encoder,
        encoder_downsample=post_encoder_downsample,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        text_encoder_dim=params.text_encoder_dim,
        joint_dim=params.joint_dim,
    )
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
    `best_train_epoch`, `best_train_loss` in `params`.

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
        "batch_idx_train",
        "best_train_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def load_model_params(
    ckpt: str, model: nn.Module, init_modules: List[str] = None, strict: bool = True
):
    """Load model params from checkpoint

    Args:
        ckpt (str): Path to the checkpoint
        model (nn.Module): model to be loaded
        init_modules (list[str]): List of modules to be initialized

    """
    logging.info(f"Loading checkpoint from {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")

    # if module list is empty, load the whole model from ckpt
    if not init_modules:
        if next(iter(checkpoint["model"])).startswith("module."):
            logging.info("Loading checkpoint saved by DDP")

            dst_state_dict = model.state_dict()
            src_state_dict = checkpoint["model"]
            for key in dst_state_dict.keys():
                src_key = "{}.{}".format("module", key)
                dst_state_dict[key] = src_state_dict.pop(src_key)
            assert len(src_state_dict) == 0
            model.load_state_dict(dst_state_dict, strict=strict)
        else:
            model.load_state_dict(checkpoint["model"], strict=strict)
    else:
        src_state_dict = checkpoint["model"]
        dst_state_dict = model.state_dict()
        for module in init_modules:
            logging.info(f"Loading parameters starting with prefix {module}")
            src_keys = [
                k for k in src_state_dict.keys() if k.startswith(module.strip() + ".")
            ]
            dst_keys = [
                k for k in dst_state_dict.keys() if k.startswith(module.strip() + ".")
            ]
            assert set(src_keys) == set(dst_keys)  # two sets should match exactly
            for key in src_keys:
                assert dst_state_dict[key].shape == src_state_dict[key].shape
                dst_state_dict[key] = src_state_dict.pop(key)

        model.load_state_dict(dst_state_dict, strict=strict)

    return None


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[Any] = None,
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
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer: RobertaTokenizer,
    criterion: nn.Module,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

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
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    text = tokenizer(
        batch["supervisions"]["text"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text = {k: v.to(device) for k, v in text.items()}

    batch_idx_train = params.batch_idx_train

    if params.freeze_encoder_steps > 0:
        freeze_encoder = batch_idx_train < params.freeze_encoder_steps
        if random.random() < 0.01 and is_training:
            logging.info(f"Step: {batch_idx_train}. Freeze encoder: {freeze_encoder}")
        if batch_idx_train == params.freeze_encoder_steps:
            logging.info(
                f"Reaching {params.freeze_encoder_steps}. Freeze encoder: {freeze_encoder}."
            )
    else:
        freeze_encoder = params.freeze_encoder

    with torch.set_grad_enabled(is_training):
        audio_features, text_features, logit_scale = model(
            audio=feature,
            audio_lens=feature_lens,
            text=text,
            freeze_audio_encoder=False,
            freeze_text_encoder=False,
        )
        loss = criterion(
            audio_features=audio_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )

    info = MetricsTracker()
    batch_size = len(batch["supervisions"]["text"])
    info["utterances"] = batch_size
    info["utt_clip_loss"] = loss.detach().cpu().item() * batch_size

    return loss, info


def evaluate(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer: RobertaTokenizer,
    valid_dl: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Run the validation process."""
    model.eval()

    metrics = {}
    num_samples = 0
    # Note: this does not scale past small eval datasets
    # all_audio_features @ all_text_features will blow up memory and compute very quickly
    eval_info = {
        "clip_loss": 0.0,
        "num_samples": 0,
        "all_audio_features": [],
        "all_text_features": [],
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            device = (
                model.device
                if isinstance(model, DDP)
                else next(model.parameters()).device
            )
            feature = batch["inputs"]
            # at entry, feature is (N, T, C)
            assert feature.ndim == 3
            feature = feature.to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            captions = [
                c.supervisions[0].custom["long_captions"][1]
                for c in batch["supervisions"]["cut"]
            ]
            text = tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text = {k: v.to(device) for k, v in text.items()}

            audio_features, text_features, logit_scale = model(
                audio=feature,
                audio_lens=feature_lens,
                text=text,
                freeze_audio_encoder=True,
                freeze_text_encoder=True,
            )

            num_samples += audio_features.shape[0]

            eval_info["all_audio_features"].append(audio_features.cpu())
            eval_info["all_text_features"].append(text_features.cpu())

            if batch_idx % 100 == 0:
                logging.info(f"Validation batch {batch_idx}")

        metrics_single_dataset = compute_metrics(
            audio_features=torch.cat(eval_info["all_audio_features"]),
            text_features=torch.cat(eval_info["all_text_features"]),
            logit_scale=logit_scale.cpu(),
        )
        metrics.update(metrics_single_dataset)

    return metrics


@torch.no_grad()
def compute_metrics(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Dict[str, float]:
    assert audio_features.dim() == 2 and text_features.dim() == 2, "Shapes must match"
    assert audio_features.shape[0] == text_features.shape[0], "Batch sizes must match"
    assert audio_features.shape[1] == text_features.shape[1], "Feature dims must match"

    N = audio_features.shape[0]

    logits_per_audio = logit_scale * audio_features @ text_features.t()
    logits_per_text = logits_per_audio.t()

    labels = torch.arange(N, dtype=torch.long)

    total_loss = (
        F.cross_entropy(logits_per_audio, labels)
        + F.cross_entropy(logits_per_text, labels)
    ) / 2

    metrics = {}
    metrics["clip_loss"] = total_loss.item()
    metrics["num_samples"] = N

    for name, logit in {
        "audio_to_text": logits_per_audio,
        "text_to_audio": logits_per_text,
    }.items():
        ranking = torch.argsort(logit, dim=1, descending=True)

        # preds = torch.where(ranking == ground_truth)[1]
        ranks = torch.empty_like(ranking)
        ranks.scatter_(1, ranking, torch.arange(N).unsqueeze(0).expand(N, -1))
        idx = torch.arange(N)
        preds = ranks[idx, idx]

        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = (preds < k).float().mean().item()

        metrics[f"{name}_mAP@10"] = (
            torch.where(
                preds < 10,
                1.0 / (preds.float() + 1.0),
                torch.zeros_like(preds, dtype=torch.float),
            )
            .mean()
            .item()
        )

    return metrics


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer: RobertaTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    criterion: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    valid_dls: torch.utils.data.DataLoader,
    valid_sets: List[str],
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
    device = torch.device("cuda", torch.cuda.current_device())

    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=0,
        )

    def slice_batch(batch, n):
        if isinstance(batch, dict):
            return {k: slice_batch(v, n) for k, v in batch.items()}
        if isinstance(batch, tuple):
            return tuple(slice_batch(v, n) for v in batch)
        if isinstance(batch, list):
            return [slice_batch(v, n) for v in batch[:n]]
        if isinstance(batch, torch.Tensor):
            if batch.dim() == 0:
                return batch
            return batch[:n]
        return batch

    train_iter = iter(train_dl)
    batch_idx = -1
    while True:
        batch_idx += 1

        try:
            batch = next(train_iter)
            batch_size = len(batch["supervisions"]["text"])
        except StopIteration:
            batch_size = 0

        if world_size > 1:
            t = torch.tensor([batch_size], dtype=torch.int64, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            min_batch_size = int(t.item())

            if min_batch_size == 0:
                batch_size = 0
            else:
                if batch_size > min_batch_size:
                    batch = slice_batch(batch, min_batch_size)
                    batch_size = min_batch_size

        if batch_size == 0:
            logging.info(f"Epoch {params.cur_epoch} finished.")
            train_dl.sampler.cuts_iter.close()
            break

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1

        try:
            with torch.cuda.amp.autocast(
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    tokenizer=tokenizer,
                    criterion=criterion,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==mean and loss is computed over utterances
            # in the batch.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

            optimizer.zero_grad()
        except Exception as e:  # noqa
            logging.warning(e)
            save_bad_model()
            display_and_save_batch(batch, params=params)
            raise e

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
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.use_autocast:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_autocast else 1.0

            cur_batch_idx = batch_idx

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {cur_batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_autocast else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if (
            0
            and batch_idx % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            if rank == 0:
                for valid_set, valid_dl in zip(valid_sets, valid_dls):
                    logging.info(f"Do validation on {valid_set}")
                    metrics = evaluate(
                        params=params,
                        model=unwrap_model(model),
                        tokenizer=tokenizer,
                        valid_dl=valid_dl,
                    )
                    model.train()
                    logging.info(
                        f"Epoch {params.cur_epoch}, "
                        f"validation on {valid_set}, "
                        + " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    )
                    logging.info(
                        f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
                    )
                    if tb_writer is not None:
                        for name, val in metrics.items():
                            tb_writer.add_scalar(
                                f"valid/{valid_set}-{name}", val, params.batch_idx_train
                            )
                    with open(
                        f"{params.exp_dir}/log/log-valid-{valid_set}.jsonl", "a+"
                    ) as f:
                        f.write(json.dumps(metrics) + "\n")

    loss_value = tot_loss["utt_clip_loss"] / tot_loss["utterances"]
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

    if params.use_multi_node:
        local_rank = get_local_rank()
    else:
        local_rank = rank
    logging.info(f"rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")

    if world_size > 1:
        setup_dist(rank, world_size, params.master_port, params.use_multi_node)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    logging.info(f"Device: {device}, rank: {rank}, local_rank: {local_rank}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    if params.use_bf16:  # amp + bf16
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:  # amp + fp16
        params.dtype = torch.float16
        params.use_autocast = True
    else:  # fp32
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(f"Using dtype={params.dtype}")
    logging.info(f"Use AMP={params.use_autocast}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    # load model parameters for model fine-tuning
    if params.do_finetune:
        assert params.start_epoch == 1, "Fine-tune must start from epoch 1"
        modules = params.init_modules.split(",") if params.init_modules else None
        checkpoints = load_model_params(
            ckpt=params.finetune_ckpt, model=model, init_modules=modules
        )
        # Need to update the model_avg if use initialisation
        if rank == 0:
            # model_avg is only used with rank 0
            compare_model(model.state_dict(), model_avg.state_dict())
            model_avg = copy.deepcopy(model).to(torch.float64)
    else:
        if params.start_epoch > 1:
            # resuming training
            checkpoints = load_checkpoint_if_available(
                params=params, model=model, model_avg=model_avg
            )
        else:
            # training from scratch
            checkpoints = None

    # Setting the encoder lr scale
    logging.info(
        f"Setting the lr scale of parameters in encoder and encoder_embed to {params.encoder_lr_scale}"
    )
    if params.encoder_lr_scale != 1.0:
        model.encoder.lr_scale = params.encoder_lr_scale
        model.encoder_embed.lr_scale = params.encoder_lr_scale
        model.text_encoder.lr_scale = params.encoder_lr_scale

    # Check the freezing encoder configuration
    if params.freeze_encoder_steps > 0:
        logging.info(f"Freeze the encoder for {params.freeze_encoder_steps} steps")
        assert not params.freeze_encoder
    if params.freeze_encoder:
        logging.info(f"Freeze the encoder for the whole training")
        assert params.freeze_encoder_steps < 0

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = get_criterion(params, world_size, rank)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(
        optimizer,
        params.lr_batches,
        params.lr_epochs,
        warmup_batches=params.warmup_batches,
        warmup_start=params.warmup_start,
    )

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

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    datamodule = DataModule(args)

    train_cuts = datamodule.emilia_en_cuts()

    def remove_short_and_long_utt(c: Any):
        # Keep only utterances with duration between 4 second and 30 seconds
        if c.duration < 4.0 or c.duration > 30.0:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if rank == 0:
        duration_bins = datamodule.estimate_duration_bins(
            cuts=train_cuts,
            world_size=world_size,
            rank=rank,
        )
        datamodule.args.duration_bins = duration_bins
        logging.info(f"Duration bins: {duration_bins}")
    if world_size > 1:
        obj_list = [duration_bins if rank == 0 else None]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        datamodule.args.duration_bins = obj_list[0]

    last_upper = 30.0
    datamodule.args.max_seq_len_buckets = datamodule.args.duration_bins + [last_upper]
    datamodule.args.fixed_batch_sizes = [
        max(1, int(params.max_duration // ub))
        for ub in datamodule.args.max_seq_len_buckets
    ]

    # construct the training dataloader
    train_dl = datamodule.train_dataloaders(
        train_cuts,
        world_size=world_size,
        rank=rank,
    )

    valid_sets = []
    valid_dls = []
    if 0 and rank == 0:
        valid_cuts = datamodule.dev_clean_cuts()
        valid_sets.append("librispeech")
        valid_dls.append(
            datamodule.valid_dataloaders(
                valid_cuts,
                world_size=1,
                rank=rank,
            )
        )

    scaler = create_grad_scaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_dl=train_dl,
            valid_dls=valid_dls,
            valid_sets=valid_sets,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

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
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    features = batch["inputs"]
    logging.info(f"features shape: {features.shape}")


def main():
    parser = get_parser()
    DataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    if args.use_multi_node:
        rank = get_rank()
        world_size = get_world_size()
        args.world_size = world_size
        run(rank=rank, world_size=world_size, args=args)
    else:
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
