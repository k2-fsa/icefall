 #!/usr/bin/env python3
#
# Copyright    2024  University of Cambridge  (authors: Xiaoyu Yang,
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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# For non-streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --full-libri 1 \
  --max-duration 1000

# For streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --causal 1 \
  --full-libri 1 \
  --max-duration 1000

It supports training with:
  - transducer loss (default), with `--use-transducer True --use-ctc False`
  - ctc loss (not recommended), with `--use-transducer False --use-ctc True`
  - transducer loss & ctc loss, with `--use-transducer True --use-ctc True`
"""


import argparse
import copy
import logging
from functools import partial
import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from kd_datamodule3_shar_speech_audio_multi_teacher import MultiTaskDataModule
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model_multi_kd_multi_teacher import MultiKDModel
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import _add_task_id, MetricsTracker, setup_distributed

from zipformer2 import Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import (
    cleanup_dist,
    get_rank,
    get_world_size,
)
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_finetune_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--do-finetune",
        type=str2bool,
        default=False,
        help="Whether to fine-tune.",
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
        "--freeze-modules",
        type=str,
        default=None,
        help="""
        Modules to be frozen. It matches all parameters starting with
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
        help="Fine-tuning from which checkpoint (a path to a .pt file)",
    )

def add_model_arguments(parser: argparse.ArgumentParser):
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

    parser.add_argument(
        "--do-mvq",
        type=str2bool,
        default=True,
    )
    
    parser.add_argument(
        "--do-audio-tagging",
        type=str2bool,
        default=True,
        help="If do audio tagging multi task training"
    )
    
    parser.add_argument(
        "--do-speaker-verification",
        type=str2bool,
        default=False,
        help="If do speaker verification"
    )    
    
    parser.add_argument(
        "--distillation-layer",
        type=str,
        default="-1,-1",
    )
    
    parser.add_argument(
        "--distillation-delta",
        type=str,
        default="0,0",
    )
    
    parser.add_argument(
        "--teacher-frame-ratio",
        type=str,
        default="2,2",
        help="The frame rate ratio between teacher and student"
    )
    
    parser.add_argument(
        "--interpolate-teacher",
        type=str2bool,
        default=False,
        help="""This should only be used when the teacher has a lower frame rate
        than the student model. We use interpolation to find the nearest neighbour"""
    )
    
    parser.add_argument(
        "--num-codebooks",
        type=str,
        default="16,16",
    )
    
    # masking related
    parser.add_argument(
        "--loss-only-mask",
        type=str2bool,
        default=False,
        help="If True, only compute loss on the masked indices"
    )
    
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="w2v2",
        choices=["w2v2", "block"],
        help="The masking mode",
    )
    
    parser.add_argument(
        "--mask-length", type=int, default=10, help="mask_length"
    )

    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.65,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        default="static",
        help="how to choose mask length",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        default=0,
        help="secondary mask argument (used for more complex distributions),see help in compute_mask_indicesh",
    )
    
    parser.add_argument(
        "--mask-channel-length", type=int, default=15, help="mask_length"
    )
    
    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        default=0.0,
        help="probability of replacing a channel with mask",
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
        default=30,
        help="Number of epochs to train.",
    )
    
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200000,
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
        default="multi_task/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=500.0
    )
    
    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
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
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)" "part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--audio-tagging-loss-scale",
        type=float,
        default=1.0,
        help="Scale for audio tagging loss.",
    )
    
    # TODO: make this applicable to more than two losses
    parser.add_argument(
        "--speech-mvq-loss-scale",
        type=float,
        default=1.0,
        help="The scale of speech mvq losses"
    )

    parser.add_argument(
        "--audio-mvq-loss-scale",
        type=float,
        default=1.0,
        help="The scale of audio mvq losses"
    )
    
    parser.add_argument(
        "--speaker-verification-loss-scale",
        type=float,
        default=1.0,
        help="Scale for audio tagging loss.",
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
        default=True,
        help="Whether to use half precision training.",
    )
    
    parser.add_argument(
        "--use-multi-node",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--stop-early",
        type=str2bool,
        default=True,
        help="If stop early if using mux"
    )

    add_finetune_arguments(parser)
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

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
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
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 128, # for better audio capability 
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
            # parameters for multitask
            "num_tasks": 2,
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))

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


def get_model(params: AttributeDict) -> nn.Module:

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    
    if params.interpolate_teacher:
        logging.warning(f"Interpolate the teacher indexes to match the length of the student")
        assert params.teacher_frame_ratio == 1
        
    if params.output_downsampling_factor == 1:
        logging.info(f"Setting the output downsample factor to 1.")
        if params.teacher_frame_ratio > 1:
            logging.warning(
                f"You are using teacher_frame_ratio={params.teacher_frame_ratio}. "
                "However, the output downsampling factor is 1. This could be wrong!"
            )
    
    assert params.enable_spec_aug == False, "Should not use specaug when using w2v2 style masking"
    if params.loss_only_mask:
        logging.info("Only computing loss on the masked positions")

    model = MultiKDModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        num_codebooks=_to_int_tuple(params.num_codebooks),
        distillation_layer=_to_int_tuple(params.distillation_layer),
        distillation_delta=_to_int_tuple(params.distillation_delta),
        interpolate_teacher=params.interpolate_teacher,
        teacher_frame_ratio=_to_int_tuple(params.teacher_frame_ratio),
        n_mels=params.feature_dim,
        mask_mode=params.mask_mode,
        mask_prob=params.mask_prob,
        mask_length=params.mask_length,
        mask_selection=params.mask_selection,
        mask_other=params.mask_other,
        mask_channel_prob=params.mask_channel_prob,
        mask_channel_length=params.mask_channel_length,
        loss_only_mask=params.loss_only_mask,
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
    assert params.start_epoch == 1
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
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

def load_model_params(
    ckpt: str, model: nn.Module, init_modules: List[str] = None, strict: bool = True
):
    """Load model params from checkpoint

    Args:
        ckpt (str): Path to the checkpoint
        model (nn.Module): model to be loaded

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
            src_keys = [k for k in src_state_dict.keys() if k.startswith(module.strip() + ".")]
            dst_keys = [k for k in dst_state_dict.keys() if k.startswith(module.strip() + ".")]
            assert set(src_keys) == set(dst_keys)  # two sets should match exactly
            for key in src_keys:
                logging.info(f"Loading {key} from init ckpt")
                dst_state_dict[key] = src_state_dict.pop(key)

        model.load_state_dict(dst_state_dict, strict=strict)

    return None



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
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
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

    supervisions = batch["supervisions"]
    cuts = supervisions["cut"]
    cut_ids = [c.id for c in cuts]
        
    feature_lens = supervisions["num_frames"].to(device)
    task_ids = batch["task_ids"].int().to(device) 
    
    if random.random() < 0.01 and is_training:
        for t in range(1, params.num_tasks+1):
            duration = sum([c.duration for c in cuts if c.task_id == t])
            logging.info(f"Number of samples from task {t}: {sum(task_ids == t).item()}/{len(task_ids)}")
            logging.info(f"Total duration of task {t}: {duration}")
    
    # mvq tokens
    mvq_tokens = batch["cb_indexes"]
    mvq_tokens = [tokens.to(device) for tokens in mvq_tokens]
    
    # audio tagging label
    if params.do_audio_tagging:
        at_targets = batch["at_targets"].to(device) # the label indices are in CED format
    else:
        at_targets = None
    
    with torch.set_grad_enabled(is_training):
        losses = model(
            x=feature,
            x_lens=feature_lens,
            codebook_indexes=mvq_tokens,
            at_targets=at_targets,
        )

        speech_mvq_loss, audio_mvq_loss = losses[:-1]
        audio_tagging_loss = losses[-1]
        loss = 0.0

        # task_id=1: ASR data
        # task_id=2: AT data
        
        # MVQ loss, first is whisper MVQ, second is Dasheng MVQ
        mvq_loss_values = []
        if params.do_mvq:
            speech_mask = task_ids == 1 # ASR data task_id=1
            num_speech_frames = feature_lens[speech_mask].sum() // 4 # equivalent frames
            if torch.isnan(speech_mvq_loss).any(): # filter the nan loss
                logging.info(f"Detected NaN in speech mvq loss")
                speech_mvq_loss = torch.nan_to_num(speech_mvq_loss, nan=0.0)
            speech_mvq_loss = (speech_mvq_loss * speech_mask).sum()
            mvq_loss_values.append(speech_mvq_loss)
            # loss += speech_mvq_loss/ (num_speech_frames + 1) * params.speech_mvq_loss_scale # TODO: make this an option
            loss += speech_mvq_loss * params.speech_mvq_loss_scale # TODO: make this an option

            audio_mask = task_ids == 2
            num_audio_frames = feature_lens[audio_mask].sum() // 4 # equivalent frames
            
            # if num_audio_frames == 0:
            #     correction_factor = 0.0
            # elif num_speech_frames == 0:
            #     correction_factor = 1.0
            # else:
            #     correction_factor = num_speech_frames / num_audio_frames
            #     if random.random() < 0.02:
            #         logging.info(f"Correction factor: {correction_factor}")
            correction_factor = 1.0
            if torch.isnan(audio_mvq_loss).any():
                logging.info(f"Detected NaN in audio mvq loss")
                audio_mvq_loss = torch.nan_to_num(audio_mvq_loss, nan=0.0)
            audio_mvq_loss = (audio_mvq_loss * audio_mask).sum()
            mvq_loss_values.append(audio_mvq_loss) # the un-normalized loss
            # loss += (audio_mvq_loss / (num_audio_frames + 1)) * params.audio_mvq_loss_scale # TODO: make this an option
            loss += audio_mvq_loss * correction_factor * params.audio_mvq_loss_scale # TODO: make this an option
        
        # AT loss
        if params.do_audio_tagging:
            mask = task_ids == 2 # AT=2
            audio_tagging_loss = (audio_tagging_loss.sum(dim=-1) * mask).sum() # this also works if mask is all False
            loss += params.audio_tagging_loss_scale * audio_tagging_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker(normalize=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
        info["utterances"] = task_ids.size(0)

    # Note: We use reduction=sum while computing the loss
    info["loss"] = loss.detach().cpu().item()
    if params.do_mvq:
        teachers = ["speech", "audio"]
        for i, mvq_loss in enumerate(mvq_loss_values):
            info[f"{teachers[i]}_mvq_loss"] = mvq_loss.detach().cpu().item()
    if params.do_audio_tagging:
        info["audio_tagging_loss"] = audio_tagging_loss.detach().cpu().item()

    # logging.info(f"Batch: {params.batch_idx_train}: speech mvq loss: {speech_mvq_loss}, num_frames: {num_speech_frames}")
    # logging.info(f"Batch: {params.batch_idx_train}: audio mvq loss: {audio_mvq_loss}, num_frames: {num_audio_frames}")
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker(normalize=True)

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
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
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_sets: List[str],
    valid_dls: List[torch.utils.data.DataLoader],
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

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler if not params.use_shar else None,
            scaler=scaler,
            rank=0,
        )
        
    def estimate_cur_epoch(max_duration: float, world_size: int, steps: int, train_hrs: int):
        estimated_hours = max_duration * world_size * steps / 3600
        estimated_epochs = estimated_hours // train_hrs
        return estimated_epochs

    shard_count = {}
    cur_epoch = 0
    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))
            
        if params.use_shar:
            est_epoch = estimate_cur_epoch(
                params.max_duration, world_size, params.batch_idx_train, params.train_duration,
            )
            if est_epoch > cur_epoch:
                cur_epoch = est_epoch
                scheduler.step_epoch(cur_epoch) # start from 1
                logging.info(f"Estimated epoch: {cur_epoch}")

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
    
        supervisions = batch["supervisions"]
        cuts = supervisions["cut"]
        
        if params.use_shar:
            shard_origin = [str(c.shard_origin).split("/")[2] for c in cuts]
            unique_origin = set(shard_origin)
            for ori in shard_origin:
                if ori in shard_count:
                    shard_count[ori] += 1
                else:
                    shard_count[ori] = 1
            count = {orig: 0 for orig in unique_origin}
            for sh in shard_origin:
                count[sh] += 1
                
            if batch_idx % 2 == 1:
                task_ids = batch["task_ids"]
                num_speech_cuts = sum(task_ids == 1).item()
                speech_duration = sum([c.duration for c in cuts if c.task_id == 1])
                num_audio_cuts = sum(task_ids == 2).item()
                audio_duration = sum([c.duration for c in cuts if c.task_id == 2])
                logging.info(f"batch {batch_idx}: task cuts: {num_speech_cuts}, {num_audio_cuts}, task durations: {speech_duration}, {audio_duration}")
                # logging.info(count)
                logging.info(f"All shards source by far: {shard_count}")
            continue
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            # save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
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
                sampler=train_dl.sampler if not params.use_shar else None,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                # if not saved_bad_model:
                #     save_bad_model(suffix="-first-warning")
                #     saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {params.batch_idx_train}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
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

        if params.batch_idx_train % params.valid_interval == 1 and not params.print_diagnostics:
            # for valid_set, valid_dl in zip(valid_sets, valid_dls):
            #     logging.info("Computing validation loss")
            #     valid_info = compute_validation_loss(
            #         params=params,
            #         model=model,
            #         sp=sp,
            #         valid_dl=valid_dl,
            #         world_size=world_size,
            #     )
            
            #     logging.info(f"Epoch {params.cur_epoch}, validation on {valid_set}: {valid_info}")
            #     logging.info(
            #         f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            #     )
            #     if tb_writer is not None:
            #         valid_info.write_summary(
            #             tb_writer, f"train/valid_{valid_set}", params.batch_idx_train
            #         )
            model.train()
        if params.use_shar and params.batch_idx_train > params.max_iters:
            return

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
    if world_size > 1:
        local_rank = setup_distributed()
    else:
        local_rank = rank

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    logging.info(f"Device: {device}")

    sp = None
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

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    # Setting the encoder lr scale
    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    parameters = get_parameter_groups_with_lrs(
        model, lr=params.base_lr, include_names=True
    )

    optimizer = ScaledAdam(
        parameters,
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs, warmup_batches=params.warmup_batches)

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

    librispeech = MultiTaskDataModule(args)

    # When using zip sampler to combine speech and audio data
    # we distribute the max-duration to each sampler according to their
    # total duration
    train_cuts = {}
    train_cuts_duration = []
    
    # NOTE: We combine all the ASR data together, and use one sampler.
    # We use CutSet.mux to sample with weight, the weight is the number 
    # of training samples (NOT the total duration)!
    asr_training_cuts = []
    asr_training_cuts_lens = []
    asr_training_cuts_duration = []
    if params.use_librispeech:
        if not params.full_libri: 
            librispeech_cuts = librispeech.train_clean_100_cuts()
            librispeech_cuts_len = 85617 # n_cuts
            librispeech_cuts_duration = 100
        else:
            librispeech_cuts = librispeech.train_all_shuf_cuts()
            librispeech_cuts_len = 281239
            librispeech_cuts_duration = 960
        librispeech_cuts = librispeech_cuts.map(partial(_add_task_id, 1)) # ASR task ID=0
        asr_training_cuts.append(librispeech_cuts)
        asr_training_cuts_lens.append(librispeech_cuts_len * params.repeat_librispeech)
        asr_training_cuts_duration.append(librispeech_cuts_duration * params.repeat_librispeech)
        
    
    if params.use_gigaspeech:
        gigaspeech_cuts = librispeech.gigaspeech_train_cuts()
        gigaspeech_cuts_len = {
            "xs": 9389,
            "s": 210012, # 250 hrs
            "m": 859493, # 1000 hrs
            "l": 2152879, # 2500 hrs
            "xl": 8611516 # 10000 hrs
        }
        gigaspeech_cuts_duration = {
            "xs": 10,
            "s": 250, # 250 hrs
            "m": 1000, # 1000 hrs
            "l": 2500, # 2500 hrs
            "xl": 10000 # 10000 hrs, avg dur 4.2s
        }
        gigaspeech_cuts = gigaspeech_cuts.map(partial(_add_task_id, 1)) # ASR task ID=1
        asr_training_cuts.append(gigaspeech_cuts)
        asr_training_cuts_lens.append(gigaspeech_cuts_len[params.gigaspeech_subset] * params.repeat_gigaspeech)
        asr_training_cuts_duration.append(gigaspeech_cuts_duration[params.gigaspeech_subset] * params.repeat_gigaspeech)
        
    if params.use_wenetspeech:
        wenetspeech_cuts = librispeech.wenetspeech_train_cuts()
        wenetspeech_cuts_len = {
            "S": 151600,
            "M": 1514500,
            "L": 13306651, # TODO: update this number
        }
        wenetspeech_cuts_duration = {
            "S": 100,
            "M": 1000,
            "L": 9700,
        }
        wenetspeech_cuts = wenetspeech_cuts.map(partial(_add_task_id, 1)) # ASR task ID=1
        asr_training_cuts.append(wenetspeech_cuts)
        asr_training_cuts_lens.append(wenetspeech_cuts_len[params.wenetspeech_subset])
        asr_training_cuts_duration.append(wenetspeech_cuts_duration[params.wenetspeech_subset])
    
    if params.use_libriheavy:
        libriheavy_cuts = librispeech.libriheavy_train_cuts()
        libriheavy_cuts_len = {
            "small": 122512 * 0.9, # 122512
            "medium": 996017, # 1093040, fewer after filtering
            "large": 10093746,
        }
        libriheavy_cuts_duration = {
            "small": 466,
            "medium": 4148,
            "large": 42074, # avg dur: 15s
        }
        libriheavy_cuts = libriheavy_cuts.map(partial(_add_task_id, 1)) # ASR task ID=1
        asr_training_cuts.append(libriheavy_cuts)
        asr_training_cuts_lens.append(libriheavy_cuts_len[params.libriheavy_subset])
        asr_training_cuts_duration.append(libriheavy_cuts_duration[params.libriheavy_subset])
    
    if params.use_mls:
        mls_cuts = librispeech.mls_train_cuts()
        mls_cuts = mls_cuts.map(partial(_add_task_id, 1))
        # mls cuts: 6000 hrs, 1409826 cuts
        asr_training_cuts.append(mls_cuts)
        asr_training_cuts_lens.append(1409826)
        asr_training_cuts_duration.append(6000)
    
    if params.use_extra_chinese_dataset:
        chineses_cuts, chinese_cut_durations, chinese_cuts_len = librispeech.multi_chinese_cuts()
        chineses_cuts = chineses_cuts.map(partial(_add_task_id, 1))
        asr_training_cuts.append(chineses_cuts)
        asr_training_cuts_lens.append(chinese_cuts_len)
        asr_training_cuts_duration.append(chinese_cut_durations)
        
    if params.use_extra_english_dataset:
        englishs_cuts, english_cut_durations, english_cuts_len = librispeech.multi_english_cuts()
        englishs_cuts = englishs_cuts.map(partial(_add_task_id, 1))
        asr_training_cuts.append(englishs_cuts)
        asr_training_cuts_lens.append(english_cuts_len)
        asr_training_cuts_duration.append(english_cut_durations)
        
    if params.use_emotion_dataset:
        other_emotion_cuts = librispeech.multi_emotion_cuts()
        msp_podcast_cuts = librispeech.msp_podcast_train_cust()
        emotion_cuts = CutSet.mux(
            *[other_emotion_cuts, msp_podcast_cuts],
            weights=[134, 52],
            stop_early=False,
        )
        emotion_cuts = emotion_cuts.map(partial(_add_task_id, 1)) # for now we treat ER cuts as part of ASR cuts
        asr_training_cuts.append(emotion_cuts)
        asr_training_cuts_lens.append(130297 * params.repeat_emo)  # 46267 + 84030
        asr_training_cuts_duration.append(186 * params.repeat_emo) # 52 + 134
        
    if params.use_fisher:
        fisher_cuts = librispeech.fisher_cuts()
        fisher_cuts = fisher_cuts.map(partial(_add_task_id, 1))
        # mls cuts: 2041 hrs, 2113438 cuts
        asr_training_cuts.append(fisher_cuts)
        asr_training_cuts_lens.append(2113438)
        asr_training_cuts_duration.append(2041)
        
    if params.use_voxpopuli:
        # multi-lingual data
        if params.voxpopuli_subset == "en_v2":
            voxpopuli_cuts = librispeech.voxpopuli_unlabelled_cuts()
            asr_training_cuts_lens.append(3059813)
            asr_training_cuts_duration.append(24151) # avg dur: 28.4
        else:
            voxpopuli_cuts = librispeech.voxpopuli_asr_train_cuts()
            asr_training_cuts_lens.append(526497)
            asr_training_cuts_duration.append(1636)
        
        voxpopuli_cuts = voxpopuli_cuts.map(partial(_add_task_id, 1))
        asr_training_cuts.append(voxpopuli_cuts)
        
    # combine the asr data into a BIG cut
    assert len(asr_training_cuts) >= 1
    if len(asr_training_cuts) >= 1:
        logging.info(f"ASR cuts: {asr_training_cuts}")
        logging.info(f"ASR cuts length: {asr_training_cuts_lens}")
        logging.info(f"ASR cuts duration: {asr_training_cuts_duration}")
        if len(asr_training_cuts) > 1:
            asr_training_cuts = CutSet.mux(
                *asr_training_cuts,
                weights=asr_training_cuts_lens,
                stop_early=False,
            )
        else:
            asr_training_cuts = asr_training_cuts[0]
    
        train_cuts["cuts_asr"] = asr_training_cuts
        train_cuts_duration.append(sum(asr_training_cuts_duration))
    
    # general audio data
    if params.do_audio_tagging:
        assert params.use_audioset, "If we do audio tagging, we must use audioset"
        
    def change_codebook_indexes(c):
        c.audio_codebook_indexes = c.codebook_indexes
        del c.codebook_indexes
        return c
        
    # audio data
    audio_training_cuts = []
    audio_training_cuts_lens = []
    audio_training_cuts_duration = []
    if params.use_audioset:
        logging.info(f"Getting audioset cuts")
        if params.repeat_audioset > 1 and not params.use_shar:
            audioset_cuts = librispeech.audioset_cuts().repeat(
                times=params.repeat_audioset,
                preserve_id=False
            )
        else:
            audioset_cuts = librispeech.audioset_cuts()
        
        audioset_cuts_lens = {
            "balanced": 21155,
            "full": 1904746,
        }
        audioset_cuts_duration = {
            "balanced": 50,
            "full": params.at_num_samples * 10 / 3600 if params.at_weighted_sampler else 5244,
        }
        audioset_cuts = audioset_cuts.map(partial(_add_task_id, 2))
        audioset_cuts = audioset_cuts.map(change_codebook_indexes)
        num_audio_cuts = audioset_cuts_lens[params.audioset_subset] * params.repeat_audioset
        audio_training_cuts.append(audioset_cuts)
        audio_training_cuts_lens.append(num_audio_cuts)
        audio_training_cuts_duration.append(audioset_cuts_duration[params.audioset_subset] * params.repeat_audioset)
        
    if params.use_music4all:
        # all 30s cuts
        music4all_cuts = librispeech.music4all_cuts() # 910 hrs, 109269 cuts
        music4all_cuts = music4all_cuts.map(partial(_add_task_id, 2))
        music4all_cuts = music4all_cuts.map(change_codebook_indexes)
        audio_training_cuts.append(music4all_cuts)
        audio_training_cuts_lens.append(109269 * params.repeat_music4all)
        audio_training_cuts_duration.append(910 * params.repeat_music4all)
        
    if params.use_vggsound:
        # all 10s cuts
        vggsound_cuts = librispeech.vggsound_train_cuts() # 427 hrs, 154142 cuts
        vggsound_cuts = vggsound_cuts.map(partial(_add_task_id, 2))
        vggsound_cuts = vggsound_cuts.map(change_codebook_indexes)
        audio_training_cuts.append(vggsound_cuts)
        audio_training_cuts_lens.append(154142 * params.repeat_vggsound)
        audio_training_cuts_duration.append(427 * params.repeat_vggsound)
        
    if params.use_bbceffect:
        # split into 10s
        bbceffect_cuts = librispeech.bbc_soundeffect_train_cuts() # 430 hrs, 160905 cuts
        bbceffect_cuts = bbceffect_cuts.map(partial(_add_task_id, 2))
        bbceffect_cuts = bbceffect_cuts.map(change_codebook_indexes)
        audio_training_cuts.append(bbceffect_cuts)
        audio_training_cuts_lens.append(160905)
        audio_training_cuts_duration.append(430)
        
    if params.use_freesound:
        # split into 10s, so all cuts <=10s
        freesound_cuts = librispeech.freesound_train_cuts() # 2811 hrs, 1028645 cuts
        freesound_cuts = freesound_cuts.map(partial(_add_task_id, 2))
        freesound_cuts = freesound_cuts.map(change_codebook_indexes)
        audio_training_cuts.append(freesound_cuts)
        audio_training_cuts_lens.append(1073093)
        audio_training_cuts_duration.append(2516)
    
    if params.use_mtg:
        # split into 10s
        mtg_cuts = librispeech.mtg_cuts() # 
        mtg_cuts = mtg_cuts.map(partial(_add_task_id, 2))
        mtg_cuts = mtg_cuts.map(change_codebook_indexes)
        audio_training_cuts.append(mtg_cuts)
        audio_training_cuts_lens.append(1032727)
        audio_training_cuts_duration.append(2812)
            
    # combine the audio datasets
    assert len(audio_training_cuts) >= 1
    if len(audio_training_cuts) >= 1:
        logging.info(f"audio cuts: {audio_training_cuts}")
        logging.info(f"audio cuts length: {audio_training_cuts_lens}")
        logging.info(f"audio cuts duration: {audio_training_cuts_duration}")
        if len(audio_training_cuts) > 1:
            audio_training_cuts = CutSet.mux(
                *audio_training_cuts,
                weights=audio_training_cuts_lens,
                stop_early=False,
            )
        else:
            audio_training_cuts = audio_training_cuts[0]
    
        train_cuts["cuts_audio"] = audio_training_cuts
        train_cuts_duration.append(sum(audio_training_cuts_duration))
        
    assert len(train_cuts) >= 1, "At least one task should be done!"
    
    logging.info(train_cuts)
    logging.info(train_cuts_duration)
    params.train_duration = sum(train_cuts_duration)
    
    def remove_short_and_long_utt(c: Cut):
        # because we have some music cuts, the duration is 30 second
        if c.duration < 0.9 or c.duration > 31.0:
            return False
        return True
    
    # If we filter the data and use weighted_sampler, the number of cuts
    # will be smaller, and won't match the sampling weight
    if not params.at_weighted_sampler:
        for k, cuts in train_cuts.items():
            train_cuts[k] = cuts.filter(remove_short_and_long_utt)
    
    # Combine the ASR and audio data together
    if params.bucketing_sampler:
        assert params.zip_sampler == False
        train_cuts = [item[1] for item in train_cuts.items()]
        if len(train_cuts) > 1:
            assert len(train_cuts) == 2, "We should only have speech and audio cuts"
            logging.info(f"Using mux to combine data")
            logging.info(f"Training cuts: {train_cuts}")
            train_cuts_lens = [sum(asr_training_cuts_lens), num_audio_cuts] 
            logging.info(f"Training cuts lens: {train_cuts_lens}")
            train_cuts = CutSet.mux(
                *train_cuts,
                weights=[2.0, 1.0],
                # weights=train_cuts_lens,
                stop_early=params.stop_early,
            )
        else:
            train_cuts = train_cuts[0]
        assert isinstance(train_cuts, CutSet), type(train_cuts)

    # NOTE: when using Shar, the sampler shouldn't have state
    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts,
        sampler_state_dict=sampler_state_dict, 
        sampling_weight=train_cuts_duration, 
        world_size=world_size,
        rank=rank,
    )

    valid_sets = []
    valid_dls = []
    
    # valid dataloaders
    if params.use_librispeech or params.use_libriheavy:
        ls_valid_cuts = librispeech.dev_clean_cuts()
        ls_valid_cuts += librispeech.dev_other_cuts()
        ls_valid_cuts = ls_valid_cuts.map(partial(_add_task_id, 1))
        asr_ls_valid_dl = librispeech.valid_dataloaders(ls_valid_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("ASR_ls")
        valid_dls.append(asr_ls_valid_dl)
        
    if params.use_gigaspeech:
        giga_dev_cuts = librispeech.gigaspeech_dev_cuts()
        giga_dev_cuts = giga_dev_cuts.map(partial(_add_task_id, 1))
        asr_giga_valid_dl = librispeech.valid_dataloaders(giga_dev_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("ASR_giga")
        valid_dls.append(asr_giga_valid_dl)
    
    if params.use_wenetspeech:
        wenet_dev_cuts = librispeech.wenetspeech_valid_cuts()
        wenet_dev_cuts = wenet_dev_cuts.map(partial(_add_task_id, 1))
        asr_wenet_valid_dl = librispeech.valid_dataloaders(wenet_dev_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("ASR_wenet")
        valid_dls.append(asr_wenet_valid_dl)
    
    if params.use_emotion_dataset:
        msp_podcast_dev_cuts = librispeech.msp_podcast_dev_cust()
        msp_podcast_dev_cuts = msp_podcast_dev_cuts.map(partial(_add_task_id, 1))
        er_msp_dev_dl = librispeech.valid_dataloaders(msp_podcast_dev_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("ER_msp_podcast")
        valid_dls.append(er_msp_dev_dl) 
        
    if params.use_voxpopuli and params.voxpopuli_subset != "en_v2":
        voxpopuli_dev_cuts = librispeech.voxpopuli_dev_cuts()
        voxpopuli_dev_cuts = voxpopuli_dev_cuts.map(partial(_add_task_id, 1))
        asr_voxpopuli_dev_dl = librispeech.valid_dataloaders(voxpopuli_dev_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("ASR_voxpopuli")
        valid_dls.append(asr_voxpopuli_dev_dl) 
    
    if params.use_audioset:
        as_eval_cuts = librispeech.audioset_eval_cuts()
        as_eval_cuts = as_eval_cuts.map(partial(_add_task_id, 2))
        as_eval_cuts = as_eval_cuts.map(change_codebook_indexes)
        at_valid_dl = librispeech.valid_dataloaders(as_eval_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("AT_as")
        valid_dls.append(at_valid_dl)
        
    if params.use_vggsound:
        vggsound_eval_cuts = librispeech.vggsound_test_cuts()
        vggsound_eval_cuts = vggsound_eval_cuts.map(partial(_add_task_id, 2))
        vggsound_eval_cuts = vggsound_eval_cuts.map(change_codebook_indexes)
        vggsound_valid_dl = librispeech.valid_dataloaders(vggsound_eval_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("AT_vggsound")
        valid_dls.append(vggsound_valid_dl)
        
    if params.use_bbceffect:
        bbc_test_cuts = librispeech.bbc_soundeffect_test_cuts()
        bbc_test_cuts = bbc_test_cuts.map(partial(_add_task_id, 2))
        bbc_test_cuts = bbc_test_cuts.map(change_codebook_indexes)
        bbc_test_dl = librispeech.valid_dataloaders(bbc_test_cuts, world_size=world_size, rank=rank,)
        valid_sets.append("AT_bbc")
        valid_dls.append(bbc_test_dl)
        
    # if params.use_freesound:
    #     freesound_test_cuts = librispeech.freesound_test_cuts()
    #     freesound_test_cuts = freesound_test_cuts.map(partial(_add_task_id, 2))
    #     freesound_test_cuts = freesound_test_cuts.map(change_codebook_indexes)
    #     freesound_test_dl = librispeech.valid_dataloaders(freesound_test_cuts, world_size=world_size, rank=rank,)
    #     valid_sets.append("AT_freesound")
    #     valid_dls.append(freesound_test_dl)

    logging.info(f"Validation sets: {valid_sets}")

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        if not params.use_shar:
            train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_sets=valid_sets,
            valid_dls=valid_dls,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        if params.batch_idx_train > params.max_iters:
            logging.info(f"Already reached the maximum iterations: {params.max_iters}")
            break
        
        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler if not params.use_shar else None,
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
    sp: spm.SentencePieceProcessor,
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

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    # y = sp.encode(supervisions["text"], out_type=int)
    # num_tokens = sum(len(i) for i in y)
    # logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
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
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    MultiTaskDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    if args.use_multi_node:
        # for multi-node multi-GPU training
        rank = get_rank()
        world_size = get_world_size()
        args.world_size = world_size
        print(f"rank: {rank}, world_size: {world_size}")
        run(rank=rank, world_size=world_size, args=args)
        return
    
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
