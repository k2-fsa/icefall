#!/usr/bin/env python3

# Copyright 2026 Nanjie Li (linanjie0820@gmail.com)
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

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import k2
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from attention_decoder import AttentionDecoderModel

# Local project modules
from datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from encoder_interface import EncoderInterface
from joiner import Joiner
from lhotse import CutSet, load_manifest_lazy

# Data utils
from lhotse.cut import Cut
from lhotse.dataset import SpecAugment
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed

# LCMA-SRT model
from model import LCMASRTModel
from optim import Eden, LRScheduler, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

# Icefall / project deps
from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


# ------------------------------
# Helpers
# ------------------------------


def _to_int_tuple(s: str) -> Tuple[int, ...]:
    return tuple(map(int, s.split(",")))


def load_cuts_lazy(manifest_paths: str, shuffle: bool = False) -> CutSet:
    paths = [path.strip() for path in re.split(r"[,;]", manifest_paths) if path.strip()]
    if not paths:
        raise ValueError("No valid manifest paths provided.")

    logging.info(f"Loading CutSets lazily from {len(paths)} manifest files.")

    combined_cuts = load_manifest_lazy(paths[0])
    for path in paths[1:]:
        cuts = load_manifest_lazy(path)
        combined_cuts = combined_cuts + cuts
    # Optionally shuffle the combined CutSet
    if shuffle:
        logging.info(f"## start to shuffle the feature ...")
        start_time = time.time()
        combined_cuts = combined_cuts.shuffle()
        elapsed = time.time() - start_time
        logging.info(f"Shuffling took {elapsed:.2f} seconds.")

    return combined_cuts


def get_adjusted_batch_count(params: AttributeDict) -> float:
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
    }
    try:
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            np.random.set_state(state["numpy"])
        except Exception:
            logging.warning("Failed to restore NumPy RNG state")
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


_RESUME_STATE_KEYS = [
    "batch_idx_train",
    "cur_epoch",
    "best_train_epoch",
    "best_train_loss",
    "best_valid_epoch",
    "best_valid_loss",
    "start_epoch",
    "start_batch",
]


def _collect_resume_state(params: AttributeDict) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for key in _RESUME_STATE_KEYS:
        if hasattr(params, key):
            state[key] = getattr(params, key)
    if hasattr(params, "rng_state"):
        state["rng_state"] = getattr(params, "rng_state")
    return state


def _build_params_payload(params: Optional[AttributeDict]) -> Optional[Dict[str, Any]]:
    if params is None:
        return None
    if hasattr(params, "items"):
        params_to_save = dict(params.items())
    else:
        params_to_save = dict(params)
    params_to_save["params"] = _collect_resume_state(params)
    return params_to_save


# ------------------------------
# Argparse & default params
# ------------------------------


def add_encoder_args(group: argparse._ArgumentGroup, prefix: str = "asr") -> None:
    """Add Zipformer encoder args for either ASR or ST with a prefix."""
    pf = prefix
    group.add_argument(
        f"--num-encoder-layers-{pf}",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )
    group.add_argument(
        f"--downsampling-factor-{pf}",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )
    group.add_argument(
        f"--feedforward-dim-{pf}",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension per stack, comma separated.",
    )
    group.add_argument(
        f"--num-heads-{pf}",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads: a single int or comma-separated list.",
    )
    group.add_argument(
        f"--encoder-dim-{pf}",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension per stack: int or comma-separated list.",
    )
    group.add_argument(
        f"--query-head-dim-{pf}",
        type=str,
        default="32",
        help="Query/key dim per head: int or comma-separated list.",
    )
    group.add_argument(
        f"--value-head-dim-{pf}",
        type=str,
        default="12",
        help="Value dim per head: int or comma-separated list.",
    )
    group.add_argument(
        f"--pos-head-dim-{pf}",
        type=str,
        default="4",
        help="Positional-encoding dim per head: int or comma-separated list.",
    )
    group.add_argument(
        f"--pos-dim-{pf}",
        type=int,
        default=48,
        help="Positional-encoding embedding dimension",
    )
    group.add_argument(
        f"--encoder-unmasked-dim-{pf}",
        type=str,
        default="192,192,256,256,256,192",
        help=(
            "Unmasked dims in encoders for dropout aug; int or CSV; must be "
            "<= corresponding encoder_dim."
        ),
    )
    group.add_argument(
        f"--cnn-module-kernel-{pf}",
        type=str,
        default="31,31,15,15,15,31",
        help="Kernel sizes in conv modules per stack: int or CSV.",
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    # Freeze ASR
    parser.add_argument("--freeze-asr", type=str2bool, default=False)
    parser.add_argument("--freeze-frontend", type=str2bool, default=False)

    # Shared frontend
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Input feature dim (must match FBank config).",
    )
    parser.add_argument("--subsampling-factor", type=int, default=4)

    # ASR encoder & head
    add_encoder_args(parser.add_argument_group("ASR encoder"), prefix="asr")
    parser.add_argument("--decoder-dim-asr", type=int, default=512)
    parser.add_argument("--joiner-dim-asr", type=int, default=512)
    parser.add_argument("--output-downsampling-factor-asr", type=int, default=2)

    # ST encoder & head (stacked on top of ASR encoder)
    add_encoder_args(parser.add_argument_group("ST encoder"), prefix="st")
    parser.add_argument("--decoder-dim-st", type=int, default=512)
    parser.add_argument("--joiner-dim-st", type=int, default=512)
    parser.add_argument("--output-downsampling-factor-st", type=int, default=2)

    # Attention decoders (optional)
    parser.add_argument("--use-attention-decoder", type=str2bool, default=False)
    parser.add_argument("--attention-decoder-dim-asr", type=int, default=512)
    parser.add_argument("--attention-decoder-dim-st", type=int, default=512)
    parser.add_argument("--attention-decoder-num-layers", type=int, default=6)
    parser.add_argument("--attention-decoder-attention-dim", type=int, default=512)
    parser.add_argument("--attention-decoder-num-heads", type=int, default=8)
    parser.add_argument("--attention-decoder-feedforward-dim", type=int, default=2048)

    # Causal / streaming
    parser.add_argument("--causal", type=str2bool, default=False)
    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz) for streaming; must be just -1 if non-causal.",
    )
    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Max left context in frames for causal training.",
    )

    # Heads toggles & losses
    parser.add_argument("--use-transducer", type=str2bool, default=True)
    parser.add_argument("--use-ctc-asr", type=str2bool, default=False)
    parser.add_argument("--use-ctc-st", type=str2bool, default=False)
    parser.add_argument("--use-cr-ctc", type=str2bool, default=False)
    parser.add_argument("--lm-scale", type=float, default=0.25)
    parser.add_argument("--am-scale", type=float, default=0.0)
    parser.add_argument("--simple-loss-scale", type=float, default=0.5)
    parser.add_argument("--ctc-loss-scale", type=float, default=0.2)
    parser.add_argument("--cr-loss-scale", type=float, default=0.2)
    parser.add_argument("--time-mask-ratio", type=float, default=2.5)
    parser.add_argument("--attention-decoder-loss-scale", type=float, default=0.8)

    # Task weights & distillation
    parser.add_argument("--task-weight-asr", type=float, default=1.0)
    parser.add_argument("--task-weight-st", type=float, default=1.0)
    parser.add_argument(
        "--self-distill-scale",
        type=float,
        default=0.0,
        help="If the model returns a distillation loss, scale it by this factor.",
    )

    # Tokenizers
    parser.add_argument(
        "--bpe-model-asr", type=str, default="data/lang_bpe_500/bpe.model"
    )
    parser.add_argument(
        "--bpe-model-st", type=str, default="data/lang_st_bpe_1k/bpe.model"
    )
    # parser.add_argument("--ast-use-asr-data", type=int, default=0)
    # parser.add_argument("--asr-use-ast-data", type=int, default=0)

    parser.add_argument(
        "--tgt-langs",
        type=str,
        default="en,zh-cn,es",
        help="Comma-separated target language tags aligned with <2xx> labels, e.g. zh-cn,ja,de",
    )

    parser.add_argument(
        "--srt-langs",
        type=str,
        default="en,zh-cn",
        help="Comma-separated target language tags aligned with <2xx> labels, e.g. zh-cn,ja,de",
    )

    parser.add_argument(
        "--enable-st",
        type=str2bool,
        default=True,
        help="Whether to enable ST branch (encoder + losses). "
        "For 1st stage ASR-only pretraining, set to 0.",
    )

    parser.add_argument(
        "--asr-moe", type=str2bool, default=True, help="ASR: enable MoE adapter."
    )
    parser.add_argument(
        "--asr-src",
        type=str2bool,
        default=True,
        help="ASR: use source language ID (in MoE routing if asr-moe, otherwise bias).",
    )
    parser.add_argument(
        "--ast-moe", type=str2bool, default=True, help="AST: enable MoE adapter."
    )
    parser.add_argument(
        "--ast-tgt",
        type=str2bool,
        default=True,
        help="AST: use target language ID (in MoE routing if ast-moe, otherwise bias).",
    )
    parser.add_argument("--num-experts-asr", type=int, default=4)
    parser.add_argument("--num-experts-ast", type=int, default=8)
    parser.add_argument("--entropy-reg-asr", type=float, default=0.0)
    parser.add_argument("--entropy-reg-ast", type=float, default=0.0)
    parser.add_argument("--temperature-asr", type=float, default=1.0)
    parser.add_argument("--temperature-ast", type=float, default=1.0)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # DDP & bookkeeping
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=12354)
    parser.add_argument("--tensorboard", type=str2bool, default=True)

    # Training schedule
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument(
        "--reset-progress-stats",
        type=str2bool,
        default=False,
        help="Set to 1 when loading weights for a new training stage; "
        "skips restoring batch/epoch counters, RNG, and sampler state.",
    )

    # Paths
    parser.add_argument("--exp-dir", type=str, default="exp")
    parser.add_argument("--baige-tb-dir", type=str, default="exp")

    # Optim & LR
    parser.add_argument("--base-lr", type=float, default=0.045)
    parser.add_argument("--lr-batches", type=float, default=7500)
    parser.add_argument("--lr-epochs", type=float, default=3.5)
    parser.add_argument("--ref-duration", type=float, default=600.0)

    # Decoder context
    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="RNN-T decoder context size: 1=bigram; 2=trigram.",
    )

    # RNNT pruning
    parser.add_argument("--prune-range-asr", type=int, default=5)
    parser.add_argument("--prune-range-st", type=int, default=10)

    # Logging & misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-diagnostics", type=str2bool, default=False)
    parser.add_argument("--skip-sanity-check", type=str2bool, default=False)
    parser.add_argument("--inf-check", type=str2bool, default=False)
    parser.add_argument(
        "--dump-moe-routing-stats",
        type=str2bool,
        default=False,
    )

    # Checkpointing
    parser.add_argument("--save-every-n", type=int, default=4000)
    parser.add_argument("--keep-last-k", type=int, default=30)
    parser.add_argument("--average-period", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--remove-st-head", type=str2bool, default=False)
    parser.add_argument(
        "--resume-optimizer-scheduler-scaler", type=str2bool, default=True
    )

    # AMP
    parser.add_argument("--use-fp16", type=str2bool, default=False)
    parser.add_argument("--use-bf16", type=str2bool, default=False)

    # Lhotse datasets
    parser.add_argument("--train-cuts-paths", type=str, default=None)
    parser.add_argument("--valid-cuts-paths", type=str, default=None)
    parser.add_argument("--utterance-min-duration", type=float, default=0.3)
    parser.add_argument("--utterance-max-duration", type=float, default=20.0)

    parser.add_argument("--use-tgt", type=str2bool, default=False)
    # SpecAug time-warp used inside model (forward)
    # parser.add_argument("--spec-aug-time-warp-factor", type=int, default=0)
    parser.add_argument("--warm-step", type=int, default=5000)
    # Expose model args
    add_model_arguments(parser)

    # DataModule args
    LibriSpeechAsrDataModule.add_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
            "feature_dim": 80,
            "subsampling_factor": 4,
            "ignore_id": -1,
            "label_smoothing": 0.1,
            "env_info": get_env_info(),
        }
    )
    return params


# ------------------------------
# Model builders (ASR/ST)
# ------------------------------


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    return Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim_asr)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )


def _build_zipformer(
    prefix: str, output_downsampling_factor, params: AttributeDict
) -> EncoderInterface:
    enc = Zipformer2(
        output_downsampling_factor=output_downsampling_factor,
        downsampling_factor=_to_int_tuple(
            getattr(params, f"downsampling_factor_{prefix}")
        ),
        num_encoder_layers=_to_int_tuple(
            getattr(params, f"num_encoder_layers_{prefix}")
        ),
        encoder_dim=_to_int_tuple(getattr(params, f"encoder_dim_{prefix}")),
        encoder_unmasked_dim=_to_int_tuple(
            getattr(params, f"encoder_unmasked_dim_{prefix}")
        ),
        query_head_dim=_to_int_tuple(getattr(params, f"query_head_dim_{prefix}")),
        pos_head_dim=_to_int_tuple(getattr(params, f"pos_head_dim_{prefix}")),
        value_head_dim=_to_int_tuple(getattr(params, f"value_head_dim_{prefix}")),
        pos_dim=getattr(params, f"pos_dim_{prefix}"),
        num_heads=_to_int_tuple(getattr(params, f"num_heads_{prefix}")),
        feedforward_dim=_to_int_tuple(getattr(params, f"feedforward_dim_{prefix}")),
        cnn_module_kernel=_to_int_tuple(getattr(params, f"cnn_module_kernel_{prefix}")),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return enc


def get_decoder(
    prefix: str, params: AttributeDict, vocab_size: int, blank_id: int
) -> nn.Module:
    return Decoder(
        vocab_size=vocab_size,
        decoder_dim=getattr(params, f"decoder_dim_{prefix}"),
        blank_id=blank_id,
        context_size=params.context_size,
    )


def get_joiner(prefix: str, params: AttributeDict, vocab_size: int) -> nn.Module:
    return Joiner(
        encoder_dim=max(_to_int_tuple(getattr(params, f"encoder_dim_{prefix}"))),
        decoder_dim=getattr(params, f"decoder_dim_{prefix}"),
        joiner_dim=getattr(params, f"joiner_dim_{prefix}"),
        vocab_size=vocab_size,
    )


def get_attention_decoder(
    prefix: str, params: AttributeDict, vocab_size: int, sos_id: int, eos_id: int
) -> nn.Module:
    return AttentionDecoderModel(
        vocab_size=vocab_size,
        decoder_dim=getattr(params, f"attention_decoder_dim_{prefix}"),
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=params.attention_decoder_attention_dim,
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_dim,
        memory_dim=max(_to_int_tuple(getattr(params, f"encoder_dim_{prefix}"))),
        sos_id=sos_id,
        eos_id=eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )


def get_model(params: AttributeDict) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    enc_asr = _build_zipformer("asr", params.output_downsampling_factor_asr, params)
    enc_st = _build_zipformer("st", params.output_downsampling_factor_st, params)

    decoder_asr = joiner_asr = None
    decoder_st = joiner_st = None
    if params.use_transducer:
        decoder_asr = get_decoder(
            "asr", params, params.vocab_size_asr, params.blank_id_asr
        )
        joiner_asr = get_joiner("asr", params, params.vocab_size_asr)
        decoder_st = get_decoder("st", params, params.vocab_size_st, params.blank_id_st)
        joiner_st = get_joiner("st", params, params.vocab_size_st)

    attention_decoder_asr = attention_decoder_st = None
    if params.use_attention_decoder:
        attention_decoder_asr = get_attention_decoder(
            "asr", params, params.vocab_size_asr, params.sos_id_asr, params.eos_id_asr
        )
        attention_decoder_st = get_attention_decoder(
            "st", params, params.vocab_size_st, params.sos_id_st, params.eos_id_st
        )

    model = LCMASRTModel(
        encoder_embed=encoder_embed,
        enc_asr=enc_asr,
        enc_st=enc_st,
        decoder_asr=decoder_asr,
        joiner_asr=joiner_asr,
        attention_decoder_asr=attention_decoder_asr,
        decoder_st=decoder_st,
        joiner_st=joiner_st,
        attention_decoder_st=attention_decoder_st,
        encoder_dim_asr=max(_to_int_tuple(params.encoder_dim_asr)),
        encoder_dim_st=max(_to_int_tuple(params.encoder_dim_st)),
        decoder_dim_asr=params.decoder_dim_asr,
        decoder_dim_st=params.decoder_dim_st,
        vocab_size_asr=params.vocab_size_asr,
        vocab_size_st=params.vocab_size_st,
        output_downsampling_factor_asr=params.output_downsampling_factor_asr,
        output_downsampling_factor_st=params.output_downsampling_factor_st,
        num_srt_langs_asr=params.num_srt_langs_asr,
        num_tgt_langs_ast=params.num_tgt_langs_ast,
        num_experts_asr=params.num_experts_asr,
        num_experts_ast=params.num_experts_ast,
        entropy_reg_asr=params.entropy_reg_asr,
        entropy_reg_ast=params.entropy_reg_ast,
        temperature_asr=params.temperature_asr,
        temperature_ast=params.temperature_ast,
        asr_moe=params.asr_moe,
        asr_src=params.asr_src,
        ast_moe=params.ast_moe,
        ast_tgt=params.ast_tgt,
        use_transducer=params.use_transducer,
        use_ctc_asr=params.use_ctc_asr,
        use_ctc_st=params.use_ctc_st,
        use_attention_decoder=params.use_attention_decoder,
        freeze_asr=params.freeze_asr,
        freeze_frontend=params.freeze_frontend,
    )
    return model


def get_spec_augment(params: AttributeDict) -> SpecAugment:
    num_frame_masks = int(10 * params.time_mask_ratio)
    max_frames_mask_fraction = 0.15 * params.time_mask_ratio
    logging.info(
        f"num_frame_masks: {num_frame_masks}, max_frames_mask_fraction: {max_frames_mask_fraction}"
    )
    return SpecAugment(
        time_warp_factor=0,
        num_frame_masks=num_frame_masks,
        features_mask_size=27,
        num_feature_masks=2,
        frames_mask_size=100,
        max_frames_mask_fraction=max_frames_mask_fraction,
    )


# ------------------------------
# Checkpoint helpers
# ------------------------------


def _load_model_weights_drop_st_head(
    ckpt_path, model, st_prefixes=None, log_drops=True
):
    import logging

    import torch

    if st_prefixes is None:
        st_prefixes = [
            "decoder_st.",
            "joiner_st.",
            "simple_am_proj_st",
            "simple_lm_proj_st",
            "ctc_st",
            "attention_decoder_st",
        ]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model"]
    tgt_model = model.module if hasattr(model, "module") else model
    tgt = tgt_model.state_dict()

    def is_st_key(k: str) -> bool:
        return any(k.startswith(p) or (p in k) for p in st_prefixes)

    keep, dropped = {}, []
    for k, v in src.items():
        if k in tgt and not is_st_key(k):
            keep[k] = v
        else:
            if is_st_key(k):
                dropped.append(k)

    missing, unexpected = tgt_model.load_state_dict(keep, strict=False)
    if log_drops and dropped:
        for k in dropped:
            logging.warning(f"[CKPT] Drop ST param unconditionally: {k}")
    rest = {
        k: v
        for k, v in ckpt.items()
        if k not in ("model", "optimizer", "scheduler", "grad_scaler")
    }
    return {"missing": missing, "unexpected": unexpected, "dropped": dropped, **rest}


def _safe_load_ckpt_with_filter(
    ckpt_path: Path,
    model: nn.Module,
    drop_prefixes: Optional[List[str]] = None,
    log_mismatch: bool = True,
) -> Dict[str, Any]:
    import logging

    import torch

    drop_prefixes = drop_prefixes or []
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model"]

    tgt_model = model.module if hasattr(model, "module") else model
    tgt_state = tgt_model.state_dict()

    keep = {}
    dropped = []

    def _should_drop(name: str) -> bool:
        return any(name.startswith(pfx) for pfx in drop_prefixes)

    for k, v in src.items():
        if _should_drop(k):
            dropped.append(
                (
                    k,
                    f"drop-by-prefix({[p for p in drop_prefixes if k.startswith(p)][0]})",
                )
            )
            continue
        if k in tgt_state and tgt_state[k].shape == v.shape:
            keep[k] = v
        else:
            reason = "shape-mismatch" if (k in tgt_state) else "missing-in-target"
            dropped.append((k, reason))

    missing, unexpected = tgt_model.load_state_dict(keep, strict=False)

    if log_mismatch:
        for k, why in dropped:
            old_shape = src[k].shape
            new_shape = tgt_state[k].shape if k in tgt_state else "N/A"
            logging.warning(f"[CKPT] Drop {k}: {why}; {old_shape} -> {new_shape}")
        if missing:
            logging.warning(
                f"[CKPT] Missing keys after load (ok with strict=False): {missing[:8]}{' ...' if len(missing)>8 else ''}"
            )
        if unexpected:
            logging.warning(
                f"[CKPT] Unexpected keys in ckpt (ignored): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}"
            )
    rest = {k: v for k, v in ckpt.items() if k != "model"}
    return rest


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    if (
        params.resume_from_checkpoint
        and params.remove_st_head
        and params.start_batch == 0
        and params.start_epoch == 1
    ):
        filename = Path(params.resume_from_checkpoint)
        assert filename.is_file(), f"{filename} does not exist!"
        info = _load_model_weights_drop_st_head(filename, model)
        if model_avg is not None:
            with torch.no_grad():
                model_avg.load_state_dict(
                    (model.module if hasattr(model, "module") else model).state_dict()
                )
        if "params" in info:
            for k in [
                "best_train_epoch",
                "best_valid_epoch",
                "batch_idx_train",
                "best_train_loss",
                "best_valid_loss",
                "cur_epoch",
            ]:
                if k in info["params"]:
                    params[k] = info["params"][k]
        return info

    elif (
        params.resume_from_checkpoint
        and params.start_batch == 0
        and params.start_epoch == 1
    ):
        filename = Path(params.resume_from_checkpoint)
    elif params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    drop_prefixes = ["ast_moe_layer.router."]

    rest = _safe_load_ckpt_with_filter(filename, model, drop_prefixes=drop_prefixes)

    if model_avg is not None:
        with torch.no_grad():
            model_avg.load_state_dict(
                (model.module if hasattr(model, "module") else model).state_dict()
            )
    saved_params: Dict[str, Any] = {}
    reset_progress = getattr(params, "reset_progress_stats", False)
    if params.resume_optimizer_scheduler_scaler and not reset_progress:
        if "optimizer" in rest and optimizer is not None:
            logging.info("Loading optimizer state dict")
            try:
                optimizer.load_state_dict(rest["optimizer"])
            except Exception as e:
                logging.warning(f"Load optimizer state failed: {e}")
        if (
            "scheduler" in rest
            and scheduler is not None
            and rest["scheduler"] is not None
        ):
            logging.info("Loading scheduler state dict")
            try:
                scheduler.load_state_dict(rest["scheduler"])
            except Exception as e:
                logging.warning(f"Load scheduler state failed: {e}")
        if "grad_scaler" in rest:
            saved_params["grad_scaler"] = rest["grad_scaler"]
    elif params.resume_optimizer_scheduler_scaler and reset_progress:
        logging.info(
            "Skipping optimizer/scheduler/scaler restoration due to --reset-progress-stats=1"
        )

    if "sampler" in rest and not reset_progress:
        saved_params["sampler"] = rest["sampler"]
    elif "sampler" in rest and reset_progress:
        logging.info(
            "Skipping sampler state restoration due to --reset-progress-stats=1"
        )

    resume_state = rest.get("params")
    if resume_state is None:
        fallback_state = {k: rest[k] for k in _RESUME_STATE_KEYS if k in rest}
        resume_state = fallback_state if fallback_state else None
        if resume_state:
            saved_params["params"] = resume_state
    else:
        saved_params["params"] = resume_state

    if resume_state:
        if not reset_progress:
            for k in [
                "best_train_epoch",
                "best_valid_epoch",
                "batch_idx_train",
                "best_train_loss",
                "best_valid_loss",
                "cur_epoch",
            ]:
                if k in resume_state:
                    params[k] = resume_state[k]

            if params.start_batch > 0 and "cur_epoch" in resume_state:
                params["start_epoch"] = resume_state["cur_epoch"]

            if "rng_state" in resume_state:
                _restore_rng_state(resume_state["rng_state"])
        else:
            logging.info(
                "reset-progress-stats=1: skipping training counters and RNG restoration."
            )

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
    if rank != 0:
        return
    params.rng_state = _capture_rng_state()
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    params_payload = _build_params_payload(params)
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params_payload,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-train-loss.pt")
    if params.best_valid_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-valid-loss.pt")


from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _normalize_lang_tag(tag: Optional[str]) -> Optional[str]:
    if tag is None:
        return None
    if not isinstance(tag, str):
        return tag
    normalized = tag.strip()
    if not normalized:
        return None
    return normalized.lower()


def build_srctgt_lang_list(src_langs: List[str], tgt_langs: List[str]) -> List[str]:
    """
    Generate human-readable labels for every (src, tgt) language pair so
    MoE routing stats can show directions like en->de or de->en.
    """
    if not src_langs or not tgt_langs:
        return []
    return [f"{src}->{tgt}" for src in src_langs for tgt in tgt_langs]


def _extract_st_texts_and_lang_ids(
    supervisions: List[Dict[str, Any]],
    use_tgt: bool,
    tgt_lang2id: Dict[str, int],
    default_lang: str = None,
):
    default_lang = _normalize_lang_tag(default_lang)
    st_texts = []
    lang_ids: list[int] = []

    for cut in supervisions["cut"]:
        for supervision in cut.supervisions:
            if hasattr(supervision, "custom") and "st_text" in supervision.custom:
                lang_tag = None
                if "lang" in supervision.custom and supervision.custom["lang"]:
                    lang_tag = _normalize_lang_tag(supervision.custom["lang"])
                if lang_tag is None and default_lang is not None:
                    lang_tag = default_lang

                if use_tgt:
                    if lang_tag is None:
                        raise ValueError(
                            "Missing custom['lang'] and no default_lang provided."
                        )
                    if lang_tag not in tgt_lang2id:
                        raise KeyError(
                            f"Unknown target language tag: {lang_tag}. "
                            f"Known: {list(tgt_lang2id.keys())}"
                        )

                    supervision.custom["st_text"] = (
                        f"<2{lang_tag}>" + supervision.custom["st_text"]
                    )
                    lang_ids.append(tgt_lang2id[lang_tag])
                else:
                    if lang_tag is None:
                        lang_ids.append(0)
                    else:
                        lang_ids.append(tgt_lang2id.get(lang_tag, 0))

                st_texts.append(supervision.custom["st_text"])
    tgt_lang_ids = torch.tensor(lang_ids, dtype=torch.long)
    return st_texts, tgt_lang_ids


def asr_source_lang_tensor(
    supervisions: Dict[str, Any],
    srt_lang2id: Dict[str, int],
    *,
    strict: bool = True,
) -> torch.LongTensor:
    tags: List[Optional[str]] = []
    cuts: Iterable[Any] = supervisions.get("cut", [])
    for cut in cuts:
        sups = getattr(cut, "supervisions", None)
        if sups is None and isinstance(cut, dict):
            sups = cut.get("supervisions", [])
        if not sups:
            continue

        for sup in sups:
            if isinstance(sup, dict):
                text = sup.get("text")
                lang = sup.get("language")
            else:
                text = getattr(sup, "text", None)
                lang = getattr(sup, "language", None)

            if text is None:
                continue

            lang = _normalize_lang_tag(lang)

            if lang is None:
                raise KeyError("Missing supervision['language'] for an ASR sample.")
            tags.append(lang)

    if "text" in supervisions and isinstance(supervisions["text"], list):
        assert len(tags) == len(
            supervisions["text"]
        ), f"The number of ASR languages ​​({len(tags)}) is inconsistent with the number of texts ({len(supervisions['text'])})."
    if strict:
        ids = []
        for t in tags:
            if t not in srt_lang2id:
                raise ValueError(
                    f"Unknown source language: {t}. Known: {list(srt_lang2id.keys())}"
                )
            ids.append(srt_lang2id[t])
    else:
        ids = [srt_lang2id.get(t, 0) for t in tags]

    return torch.tensor(ids, dtype=torch.long)


def _create_moe_stat_buffers(num_experts: int):
    if num_experts <= 0:
        return None, None
    return (
        defaultdict(lambda: torch.zeros(num_experts, dtype=torch.float64)),
        defaultdict(int),
    )


def _accumulate_moe_stats(storage, counts, batch_info):
    if storage is None or counts is None or batch_info is None:
        return
    lang_ids, weights = batch_info
    if lang_ids is None or weights is None:
        return
    lang_list = lang_ids.tolist()
    for idx, w in zip(lang_list, weights):
        storage[idx] += w.to(storage[idx].dtype)
        counts[idx] += 1


def _log_moe_stats(
    task_name: str,
    storage,
    counts,
    lang_list: List[str],
    global_step: int,
    tb_writer: Optional[SummaryWriter] = None,
):
    if storage is None or counts is None or not storage:
        return
    logging.info("===== MoE routing stats (%s) =====", task_name)
    for lang_id in sorted(storage.keys()):
        count = counts[lang_id]
        if count == 0:
            continue
        avg = storage[lang_id] / count
        lang = lang_list[lang_id] if 0 <= lang_id < len(lang_list) else str(lang_id)
        dist = ", ".join(f"e{i}:{float(val):.3f}" for i, val in enumerate(avg))
        logging.info("  %s (id=%d, n=%d): %s", lang, lang_id, count, dist)
        if tb_writer is not None:
            for expert_idx, val in enumerate(avg):
                tb_writer.add_scalar(
                    f"moe/{task_name}/{lang}/expert_{expert_idx}",
                    float(val),
                    global_step,
                )


def _reset_moe_stats(storage, counts):
    if storage is not None:
        storage.clear()
    if counts is not None:
        counts.clear()


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp_asr: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    spec_augment: Optional[SpecAugment] = None,
) -> Tuple[Tensor, MetricsTracker]:
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    feature = batch["inputs"].to(device)  # (N, T, C)
    assert feature.ndim == 3
    use_tgt = params.use_tgt
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    collect_moe_stats = bool(getattr(params, "dump_moe_routing_stats", False))

    texts_asr: List[str] = supervisions["text"]
    if params.asr_src:
        srt_lang_ids = asr_source_lang_tensor(
            supervisions, params.srt_lang2id, strict=True
        )
    else:
        srt_lang_ids = None
    stats_srt_lang_ids = srt_lang_ids

    if params.enable_st:
        texts_st, tgt_lang_ids = _extract_st_texts_and_lang_ids(
            supervisions, use_tgt, params.tgt_lang2id
        )
    else:
        texts_st, tgt_lang_ids = [], None

    stats_tgt_lang_ids = tgt_lang_ids
    if collect_moe_stats and params.use_cr_ctc:
        if stats_srt_lang_ids is not None:
            stats_srt_lang_ids = stats_srt_lang_ids.repeat(2)
        if stats_tgt_lang_ids is not None:
            stats_tgt_lang_ids = stats_tgt_lang_ids.repeat(2)

    speace_id = sp_st.piece_to_id("▁")
    y_asr = k2.RaggedTensor(sp_asr.encode(texts_asr, out_type=int))

    if params.enable_st:
        if use_tgt:
            y_st_sp_list = sp_st.encode(texts_st, out_type=int)
            y_st_sp = [
                ids[1:] if ids and ids[0] == speace_id else ids for ids in y_st_sp_list
            ]
            y_st = k2.RaggedTensor(y_st_sp)
        else:
            y_st = k2.RaggedTensor(sp_st.encode(texts_st, out_type=int))
    else:
        y_st = None

    # SpecAug/CR-CTC supports supervision segments
    use_cr_ctc = params.use_cr_ctc
    use_spec_aug = use_cr_ctc and is_training
    if use_spec_aug:
        sv = supervisions
        supervision_segments = torch.stack(
            [sv["sequence_idx"], sv["start_frame"], sv["num_frames"]], dim=1
        )
    else:
        supervision_segments = None

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    with torch.set_grad_enabled(is_training):
        outputs = model(
            x=feature,
            x_lens=feature_lens,
            y_asr=y_asr,
            y_st=y_st,
            srt_lang_ids=srt_lang_ids,
            tgt_lang_ids=tgt_lang_ids,
            prune_range_asr=params.prune_range_asr,
            prune_range_st=params.prune_range_st,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            use_cr_ctc=use_cr_ctc,
            use_spec_aug=use_spec_aug,
            spec_augment=spec_augment,
            supervision_segments=supervision_segments,
            time_warp_factor=params.spec_aug_time_warp_factor,
            enable_st=params.enable_st,
        )

        # Expected common return (allow extra tails):
        # simple_asr, simple_st, pruned_asr, pruned_st, ctc_asr, ctc_st,
        # attn_asr, attn_st, cr_asr, cr_st, [optional distill]
        (
            simple_asr,
            simple_st,
            pruned_asr,
            pruned_st,
            ctc_asr,
            ctc_st,
            attn_asr,
            attn_st,
            cr_asr,
            cr_st,
            moe_ent_loss,
            *rest,
        ) = outputs

        # distill = rest[0] if (len(rest) > 0 and isinstance(rest[0], torch.Tensor)) else None

        loss = torch.tensor(0.0, device=feature.device)
        loss_asr = torch.tensor(0.0, device=feature.device)
        loss_st = torch.tensor(0.0, device=feature.device)

        # Transducer head (simple + pruned)
        if params.use_transducer:

            def _warm_scale(s: float) -> Tuple[float, float]:
                simple_scale = (
                    s
                    if batch_idx_train >= warm_step
                    else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
                )
                pruned_scale = (
                    1.0
                    if batch_idx_train >= warm_step
                    else 0.1 + 0.9 * (batch_idx_train / warm_step)
                )
                return simple_scale, pruned_scale

            s_asr_scale, p_asr_scale = _warm_scale(params.simple_loss_scale)
            loss_asr += s_asr_scale * simple_asr + p_asr_scale * pruned_asr

            if params.enable_st:
                s_st_scale, p_st_scale = _warm_scale(params.simple_loss_scale)
                loss_st += s_st_scale * simple_st + p_st_scale * pruned_st
            # loss = loss + params.task_weight_asr * (s_asr * simple_asr + p_asr * pruned_asr)
            # loss = loss + params.task_weight_st  * (s_st  * simple_st  + p_st  * pruned_st)

        # CTC (with optional CR-CTC)
        if params.use_ctc_asr:
            # loss = loss + params.task_weight_asr * (params.ctc_loss_scale * ctc_asr)
            # loss = loss + params.task_weight_st  * (params.ctc_loss_scale * ctc_st)
            loss_asr += params.ctc_loss_scale * ctc_asr  # ctc_loss_scale default 0.2
            if params.use_cr_ctc:
                # loss = loss + params.task_weight_asr * (params.cr_loss_scale * cr_asr)
                # loss = loss + params.task_weight_st  * (params.cr_loss_scale * cr_st)
                loss_asr += params.cr_loss_scale * cr_asr  # cr_loss_scale default 0.2

        if params.use_ctc_st and params.enable_st:
            # loss = loss + params.task_weight_asr * (params.ctc_loss_scale * ctc_asr)
            # loss = loss + params.task_weight_st  * (params.ctc_loss_scale * ctc_st)
            loss_st += params.ctc_loss_scale * ctc_st
            if params.use_cr_ctc:
                # loss = loss + params.task_weight_asr * (params.cr_loss_scale * cr_asr)
                # loss = loss + params.task_weight_st  * (params.cr_loss_scale * cr_st)
                loss_st += params.cr_loss_scale * cr_st

        # Attention decoder
        if params.use_attention_decoder:
            # loss = loss + params.task_weight_asr * (params.attention_decoder_loss_scale * attn_asr)
            # loss = loss + params.task_weight_st  * (params.attention_decoder_loss_scale * attn_st)
            loss_asr += params.attention_decoder_loss_scale * attn_asr
            if params.enable_st:
                loss_st += params.attention_decoder_loss_scale * attn_st

        # Optional self-distillation term
        # if distill is not None and params.self_distill_scale > 0:
        # loss = loss + params.self_distill_scale * distill

    # loss = params.task_weight_asr * loss_asr + params.task_weight_st * loss_st + moe_ent_loss
    if params.enable_st:
        loss = (
            params.task_weight_asr * loss_asr
            + params.task_weight_st * loss_st
            + moe_ent_loss
        )
    else:
        loss = params.task_weight_asr * loss_asr + moe_ent_loss

    assert loss.requires_grad == is_training

    # Metrics
    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    info["loss"] = loss.detach().cpu().item()
    info["loss_asr"] = loss_asr.detach().cpu().item()
    info["loss_ast"] = loss_st.detach().cpu().item()
    info["moe_ent_loss"] = moe_ent_loss.detach().cpu().item()
    if params.use_transducer:
        info["simple_asr"] = simple_asr.detach().cpu().item()
        info["pruned_asr"] = pruned_asr.detach().cpu().item()

    if params.use_transducer and params.enable_st:
        info["simple_ast"] = simple_st.detach().cpu().item()
        info["pruned_ast"] = pruned_st.detach().cpu().item()

    if params.use_ctc_asr:
        info["ctc_asr"] = ctc_asr.detach().cpu().item()
        if params.use_cr_ctc:
            info["cr_asr"] = cr_asr.detach().cpu().item()
    # if params.use_ctc_st:
    #     info["ctc_st"] = ctc_st.detach().cpu().item()
    #     if params.use_cr_ctc:
    #         info["cr_st"] = cr_st.detach().cpu().item()
    if params.use_ctc_st and params.enable_st:
        info["ctc_ast"] = ctc_st.detach().cpu().item()
        if params.use_cr_ctc:
            info["cr_ast"] = cr_st.detach().cpu().item()

    if params.use_attention_decoder:
        info["attn_asr"] = attn_asr.detach().cpu().item()

    if params.use_attention_decoder and params.enable_st:
        info["attn_ast"] = attn_st.detach().cpu().item()
        # info["attn_st"] = attn_st.detach().cpu().item()
    # if distill is not None:
    #     info["distill"] = distill.detach().cpu().item()

    moe_batch_stats = None
    if collect_moe_stats:
        moe_batch_stats = {}
        base_model = model.module if isinstance(model, DDP) else model
        if hasattr(base_model, "asr_moe_layer"):
            weights_asr = getattr(base_model.asr_moe_layer, "last_router_weights", None)
            if (
                weights_asr is not None
                and stats_srt_lang_ids is not None
                and weights_asr.ndim == 3
            ):
                mean_asr = weights_asr.detach().mean(dim=0).to(torch.float64).cpu()
                if mean_asr.size(0) == stats_srt_lang_ids.numel():
                    moe_batch_stats["asr"] = (
                        stats_srt_lang_ids.detach().cpu(),
                        mean_asr,
                    )
                else:
                    logging.warning(
                        f"({mean_asr.size(0)} vs {stats_srt_lang_ids.numel()})"
                    )
        if params.enable_st and hasattr(base_model, "ast_moe_layer"):
            weights_st = getattr(base_model.ast_moe_layer, "last_router_weights", None)
            if (
                weights_st is not None
                and stats_tgt_lang_ids is not None
                and weights_st.ndim == 3
            ):
                mean_st = weights_st.detach().mean(dim=0).to(torch.float64).cpu()
                if mean_st.size(0) == stats_tgt_lang_ids.numel():
                    moe_batch_stats["st"] = (
                        stats_tgt_lang_ids.detach().cpu(),
                        mean_st,
                    )
                else:
                    logging.warning(
                        f"({mean_st.size(0)} vs {stats_tgt_lang_ids.numel()})"
                    )
        if not moe_batch_stats:
            moe_batch_stats = None

    return loss, info, moe_batch_stats


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp_asr: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    if dist.is_initialized() and dist.get_rank() != 0:
        return MetricsTracker()

    model.eval()
    tot_loss = MetricsTracker()
    collect_moe_stats = bool(getattr(params, "dump_moe_routing_stats", False))
    st_lang_labels = params.tgt_lang_list
    moe_stats_asr = moe_counts_asr = moe_stats_st = moe_counts_st = None
    if collect_moe_stats:
        base_model = model.module if isinstance(model, DDP) else model
        asr_experts = getattr(
            getattr(base_model, "asr_moe_layer", None), "num_experts", 0
        )
        ast_experts = getattr(
            getattr(base_model, "ast_moe_layer", None), "num_experts", 0
        )
        moe_stats_asr, moe_counts_asr = _create_moe_stat_buffers(asr_experts)
        moe_stats_st, moe_counts_st = _create_moe_stat_buffers(ast_experts)
    for batch_idx, batch in enumerate(valid_dl):
        loss, info, batch_moe_stats = compute_loss(
            params=params,
            model=model,
            sp_asr=sp_asr,
            sp_st=sp_st,
            batch=batch,
            is_training=False,
        )
        if collect_moe_stats and batch_moe_stats:
            _accumulate_moe_stats(
                moe_stats_asr,
                moe_counts_asr,
                batch_moe_stats.get("asr"),
            )
            _accumulate_moe_stats(
                moe_stats_st,
                moe_counts_st,
                batch_moe_stats.get("st"),
            )
        assert loss.requires_grad is False
        tot_loss = tot_loss + info

    loss_value = tot_loss["loss"] / max(tot_loss["frames"], 1)
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value
    if collect_moe_stats:
        _log_moe_stats(
            "valid_ASR",
            moe_stats_asr,
            moe_counts_asr,
            params.srt_lang_list,
            params.batch_idx_train,
            None,
        )
        _log_moe_stats(
            "valid_ST",
            moe_stats_st,
            moe_counts_st,
            st_lang_labels,
            params.batch_idx_train,
            None,
        )
        _reset_moe_stats(moe_stats_asr, moe_counts_asr)
        _reset_moe_stats(moe_stats_st, moe_counts_st)
    return tot_loss


def _reapply_freeze_asr(model):
    m = model.module if isinstance(model, DDP) else model
    if hasattr(m, "freeze_asr") and m.freeze_asr and hasattr(m, "_apply_freeze_asr"):
        m._apply_freeze_asr()


# ------------------------------
# Training loop
# ------------------------------


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp_asr: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    spec_augment: Optional[SpecAugment] = None,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    model.train()
    _reapply_freeze_asr(model)
    tot_loss = MetricsTracker()
    saved_bad_model = False
    collect_moe_stats = (
        bool(getattr(params, "dump_moe_routing_stats", False)) and rank == 0
    )
    st_lang_labels = params.tgt_lang_list
    moe_stats_asr = moe_counts_asr = moe_stats_st = moe_counts_st = None
    if collect_moe_stats:
        base_model = model.module if isinstance(model, DDP) else model
        asr_experts = getattr(
            getattr(base_model, "asr_moe_layer", None), "num_experts", 0
        )
        ast_experts = getattr(
            getattr(base_model, "ast_moe_layer", None), "num_experts", 0
        )
        moe_stats_asr, moe_counts_asr = _create_moe_stat_buffers(asr_experts)
        moe_stats_st, moe_counts_st = _create_moe_stat_buffers(ast_experts)

    def save_bad_model(suffix: str = ""):
        params.rng_state = _capture_rng_state()
        params_payload = _build_params_payload(params)
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model.module if isinstance(model, DDP) else model,
            model_avg=model_avg,
            params=params_payload,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch.cuda.amp.autocast(
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, loss_info, batch_moe_stats = compute_loss(
                    params=params,
                    model=model,
                    sp_asr=sp_asr,
                    sp_st=sp_st,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                )
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info
            if collect_moe_stats and batch_moe_stats:
                _accumulate_moe_stats(
                    moe_stats_asr,
                    moe_counts_asr,
                    batch_moe_stats.get("asr"),
                )
                _accumulate_moe_stats(
                    moe_stats_st,
                    moe_counts_st,
                    batch_moe_stats.get("st"),
                )

            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params, sp_asr=sp_asr, sp_st=sp_st)
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
                model_cur=model.module if isinstance(model, DDP) else model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            params.rng_state = _capture_rng_state()
            params_payload = _build_params_payload(params)
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model.module if isinstance(model, DDP) else model,
                model_avg=model_avg,
                params=params_payload,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir, topk=params.keep_last_k, rank=rank
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
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, loss[{loss_info}], tot_loss[{tot_loss}], "
                f"batch size: {batch_size}, lr: {cur_lr:.2e}, "
                + (f"grad_scale: {cur_grad_scale}" if params.use_autocast else "")
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
            if collect_moe_stats:
                _log_moe_stats(
                    "train_ASR",
                    moe_stats_asr,
                    moe_counts_asr,
                    params.srt_lang_list,
                    params.batch_idx_train,
                    tb_writer,
                )
                _log_moe_stats(
                    "train_ST",
                    moe_stats_st,
                    moe_counts_st,
                    st_lang_labels,
                    params.batch_idx_train,
                    tb_writer,
                )
                _reset_moe_stats(moe_stats_asr, moe_counts_asr)
                _reset_moe_stats(moe_stats_st, moe_counts_st)

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp_asr=sp_asr,
                sp_st=sp_st,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            _reapply_freeze_asr(model)  # Freeze ASR
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / max(tot_loss["frames"], 1)
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    if collect_moe_stats:
        _log_moe_stats(
            "train_ASR",
            moe_stats_asr,
            moe_counts_asr,
            params.srt_lang_list,
            params.batch_idx_train,
            tb_writer,
        )
        _log_moe_stats(
            "train_ST",
            moe_stats_st,
            moe_counts_st,
            st_lang_labels,
            params.batch_idx_train,
            tb_writer,
        )
        _reset_moe_stats(moe_stats_asr, moe_counts_asr)
        _reset_moe_stats(moe_stats_st, moe_counts_st)


# ------------------------------
# Misc helpers
# ------------------------------


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp_asr: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
) -> None:
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)
    use_tgt = params.use_tgt
    supervisions = batch["supervisions"]
    features = batch["inputs"]
    logging.info(f"features shape: {features.shape}")

    texts: List[str] = supervisions["text"]
    if params.asr_src:
        srt_lang_ids = asr_source_lang_tensor(
            supervisions, params.srt_lang2id, strict=True
        )
    else:
        srt_lang_ids = None

    if params.enable_st:
        texts_st, tgt_lang_ids = _extract_st_texts_and_lang_ids(
            supervisions, use_tgt, params.tgt_lang2id
        )
    else:
        texts_st, tgt_lang_ids = [], None

    print("---" * 15)
    print("DEBUG: Displaying text from the failing batch:")
    if texts:
        print(f"  ASR Text[0]: {texts[0]}")
        if srt_lang_ids is not None and len(srt_lang_ids):
            print(f"  ASR  LangID[0]: {int(srt_lang_ids[0])}")
    if texts_st:
        print(f"  ST  Text[0]: {texts_st[0]}")
        if len(tgt_lang_ids):
            print(f"  ST  LangID[0]: {int(tgt_lang_ids[0])}")
    print("---" * 15, flush=True)

    y_asr = sp_asr.encode(texts, out_type=int)
    y_st = sp_st.encode(texts_st, out_type=int)
    num_tokens = sum(len(i) for i in y_asr) + sum(len(i) for i in y_st)
    logging.info(f"num tokens (ASR+ST): {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp_asr: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    params: AttributeDict,
    spec_augment: Optional[SpecAugment] = None,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, _, _ = compute_loss(
                    params=params,
                    model=model,
                    sp_asr=sp_asr,
                    sp_st=sp_st,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current max_duration setting. "
                    "Decrease max_duration and try again.\n"
                    f"Failing criterion: {criterion} (= {crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp_asr=sp_asr, sp_st=sp_st)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


# ------------------------------
# Entry point (multi-GPU safe)
# ------------------------------


def get_rank_info() -> Tuple[int, int]:
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ and "SLURM_LOCALID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = 0
        local_rank = 0
    return rank, local_rank


def setup_cuda_device(local_rank: int) -> None:
    device_count = torch.cuda.device_count()
    if device_count == 1:
        torch.cuda.set_device(0)
    elif local_rank < device_count:
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError(
            f"[setup_cuda_device] local_rank={local_rank} exceeds visible CUDA devices ({device_count})"
        )
    print(f"[setup_cuda_device] Using CUDA:{torch.cuda.current_device()}")


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    params = get_params()
    params.update(vars(args))
    if rank != 0:
        params.dump_moe_routing_stats = False

    # Alias encoder param names for convenience (ASR/ST)
    # Convert CLI into attribute names used by builders
    for name in (
        "num_encoder_layers",
        "downsampling_factor",
        "feedforward_dim",
        "num_heads",
        "encoder_dim",
        "query_head_dim",
        "value_head_dim",
        "pos_head_dim",
        "pos_dim",
        "encoder_unmasked_dim",
        "cnn_module_kernel",
    ):
        setattr(params, f"{name}_asr", getattr(params, f"{name}_asr"))
        setattr(params, f"{name}_st", getattr(params, f"{name}_st"))

    fix_random_seed(params.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        setup_dist(
            rank=rank,
            world_size=world_size,
            master_port=params.master_port,
        )

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    tb_writer = (
        SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
        if args.tensorboard and rank == 0
        else None
    )

    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logging.info(f"Device: {device}")

    # Tokenizers
    sp_asr = spm.SentencePieceProcessor()
    sp_asr.load(params.bpe_model_asr)
    sp_st = spm.SentencePieceProcessor()
    sp_st.load(params.bpe_model_st)

    # Ids and vocab sizes per task
    params.blank_id_asr = sp_asr.piece_to_id("<blk>")
    params.sos_id_asr = params.eos_id_asr = sp_asr.piece_to_id("<sos/eos>")
    params.vocab_size_asr = sp_asr.get_piece_size()

    params.blank_id_st = sp_st.piece_to_id("<blk>")
    params.sos_id_st = params.eos_id_st = sp_st.piece_to_id("<sos/eos>")
    params.vocab_size_st = sp_st.get_piece_size()

    # sp_st = SentencePieceProcessor() already loaded
    params.tgt_lang_list = [s.strip() for s in params.tgt_langs.split(",") if s.strip()]
    params.tgt_lang2id = {lg: i for i, lg in enumerate(params.tgt_lang_list)}
    params.num_tgt_langs_ast = len(params.tgt_lang_list)

    params.srt_lang_list = [s.strip() for s in params.srt_langs.split(",") if s.strip()]
    params.srt_lang2id = {lg: i for i, lg in enumerate(params.srt_lang_list)}
    params.num_srt_langs_asr = len(params.srt_lang_list)
    params.srctgt_lang_list = build_srctgt_lang_list(
        params.srt_lang_list, params.tgt_lang_list
    )
    # print(f"num_srt_langs_asr:{params.num_srt_langs_asr}")

    # AMP dtype
    if params.use_bf16:
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "Use either fp16 or bf16, not both."
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:
        params.dtype = torch.float16
        params.use_autocast = True
    else:
        params.dtype = torch.float32
        params.use_autocast = False
    logging.info(f"Using dtype={params.dtype}; AMP={params.use_autocast}")

    logging.info(params)
    logging.info("About to create model")
    model = get_model(params)
    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    if params.use_cr_ctc:
        assert params.use_ctc_asr
        assert not params.enable_spec_aug  # we will do spec_augment in model.py
        spec_augment = get_spec_augment(params)
    else:
        spec_augment = None

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0

    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )
    _reapply_freeze_asr(model)  # Freeze ASR
    # logging.info(f"{model}")
    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,
        clipping_scale=2.0,
    )
    scheduler: LRSchedulerType = Eden(
        optimizer, params.lr_batches, params.lr_epochs, warmup_start=0.1
    )

    if params.resume_optimizer_scheduler_scaler:
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
        opts = diagnostics.TensorDiagnosticOptions(512)
        diagnostic = diagnostics.attach_diagnostics(model, opts)
    if params.inf_check:
        register_inf_check_hooks(model)

    # Data
    librispeech = LibriSpeechAsrDataModule(args)

    if params.full_libri:
        if args.train_cuts_paths:
            train_cuts = load_cuts_lazy(args.train_cuts_paths, shuffle=True)
        else:
            train_cuts = librispeech.train_all_shuf_cuts()
    else:
        train_cuts = librispeech.train_clean_100_cuts()

    def remove_short_and_long_utt(c: Cut) -> bool:
        if (
            c.duration < args.utterance_min_duration
            or c.duration > args.utterance_max_duration
        ):
            return False
        if getattr(c, "num_frames", None) is None:
            print(
                f"WARNING: Exclude cut {c.id}. num_frames is None (features not attached).",
                file=sys.stderr,
                flush=True,
            )
            return False
        supervision = c.supervisions[0]
        custom = getattr(supervision, "custom", {}) or {}

        text = getattr(supervision, "text", None)
        if not text or not text.strip():
            print(
                f"WARNING: Exclude cut {c.id}. Missing or empty text field.",
                file=sys.stderr,
                flush=True,
            )
            return False

        st_text = custom.get("st_text")
        if not st_text or not st_text.strip():
            print(
                f"WARNING: Exclude cut {c.id}. Missing or empty st_text field.",
                file=sys.stderr,
                flush=True,
            )
            return False

        language = getattr(supervision, "language", None)
        if not language or not language.strip():
            print(
                f"WARNING: Exclude cut {c.id}. Missing or empty language field.",
                file=sys.stderr,
                flush=True,
            )
            return False

        lang = custom.get("lang")
        if not lang or not lang.strip():
            print(
                f"WARNING: Exclude cut {c.id}. Missing or empty lang field.",
                file=sys.stderr,
                flush=True,
            )
            return False
        T = ((c.num_frames - 7) // 2 + 1) // 2

        asr_tokens = sp_asr.encode(text, out_type=str)
        asr_len = len(asr_tokens)

        st_tokens = sp_st.encode(st_text, out_type=str)
        st_len = len(st_tokens)

        need = max(asr_len, st_len)
        if T < need:
            print(
                f"WARNING: Exclude cut {c.id}. Frames(after)={T}. "
                f"ASR tokens={asr_len}, ST tokens={st_len}",
                file=sys.stderr,
                flush=True,
            )
            return False
        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    resume_mid_epoch = params.start_batch > 0 and sampler_state_dict is not None
    skipped_initial_sampler_epoch = False

    if args.valid_cuts_paths:
        valid_cuts = load_cuts_lazy(args.valid_cuts_paths, shuffle=False)
    else:
        valid_cuts = librispeech.dev_clean_cuts()
        valid_cuts += librispeech.dev_other_cuts()

    valid_dl = librispeech.valid_dataloaders_gpt41(valid_cuts)

    # if not params.skip_sanity_check and not params.print_diagnostics:
    #     scan_pessimistic_batches_for_oom(
    #         model=model,
    #         train_dl=train_dl,
    #         optimizer=optimizer,
    #         sp_asr=sp_asr,
    #         sp_st=sp_st,
    #         params=params,
    #         spec_augment=spec_augment,
    #     )

    scaler = GradScaler(enabled=params.use_autocast, init_scale=1.0)
    if params.resume_optimizer_scheduler_scaler:
        if checkpoints and "grad_scaler" in checkpoints:
            logging.info("Loading grad scaler state dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        skip_sampler_epoch = (
            resume_mid_epoch
            and not skipped_initial_sampler_epoch
            and epoch == params.start_epoch
        )
        if skip_sampler_epoch:
            logging.info(
                "Skipping sampler.set_epoch for resumed epoch; sampler state restored from checkpoint."
            )
            skipped_initial_sampler_epoch = True
        else:
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
            sp_asr=sp_asr,
            sp_st=sp_st,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            spec_augment=spec_augment,
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


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    logging.info("## get rank information ...")
    rank, local_rank = get_rank_info()
    world_size = args.world_size

    setup_cuda_device(local_rank)
    run(rank=rank, world_size=world_size, args=args)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
