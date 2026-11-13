#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
#              2024  Yuekai Zhang
#              2025  Yifan  Yang
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
"""Train SPEAR-Qwen3 with autoregressive cross-entropy on LibriSpeech.

Run from egs/librispeech/ASR after preparing the cuts:

  torchrun --standalone --nproc_per_node=8 spear_qwen_ce/train.py \
    --exp-dir spear_qwen_ce/exp_0.6b \
    --llm-path-or-name download/Qwen3-0.6B \
    --speech-encoder-path-or-name download/spear-xlarge-speech-audio-v2 \
    --full-libri True --max-duration 200 --max-steps 50000

The default recipe freezes SPEAR and trains the projector and Qwen3 LoRA
adapters using BF16, ZeRO-1 and token-normalized cross-entropy. Optimizer steps,
not epochs or microbatches, determine the schedule and checkpoint cadence.
"""

import argparse
import json
import logging
import re
import warnings
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Optional, Sequence, Tuple

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from asr_datamodule import LibriSpeechAsrDataModule
from encoder import load_spear_encoder, spear_encoder_dim
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from model import DEFAULT_SPEECH_TOKEN, IGNORE_TOKEN_ID, SPEECH_LLM, EncoderProjector
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from icefall.dist import get_local_rank, get_rank, get_world_size
from icefall.env import get_env_info
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool

CE_RECIPE_VERSION = 2


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--llm-path-or-name",
        type=str,
        default="download/Qwen3-0.6B",
        help="Path or name of the large language model.",
    )

    parser.add_argument(
        "--speech-encoder-path-or-name",
        type=str,
        default="download/spear-xlarge-speech-audio-v2",
        help="Path to the SPEAR speech encoder (a local trust_remote_code HF "
        "snapshot).",
    )

    parser.add_argument(
        "--encoder-projector-ds-rate",
        type=int,
        default=5,
        help="Downsample rate for the encoder projector. SPEAR emits ~50 frames"
        "/s, so a rate of 5 yields ~10 speech tokens per second.",
    )
    parser.add_argument(
        "--use-flash-attn",
        type=str2bool,
        default=True,
        help="Whether to use flash attention for the LLM.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        type=str2bool,
        default=False,
        help="Recompute LLM activations during backward to reduce memory use.",
    )

    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=True,
        help="Whether to use lora to fine-tune the LLM (requires `peft`).",
    )

    parser.add_argument(
        "--unfreeze-llm",
        type=str2bool,
        default=True,
        help="Whether to unfreeze the LLM during training (full fine-tune when "
        "--use-lora is False).",
    )

    parser.add_argument(
        "--unfreeze-encoder",
        type=str2bool,
        default=False,
        help="Whether to fine-tune the SPEAR speech encoder instead of keeping it frozen.",
    )


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
        "--exp-dir",
        type=str,
        default="spear_qwen_ce/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="""The path to the pretrained model if it is not None. Training will
        start from this model. e.g. ./spear_qwen_ce/exp/checkpoint-2000/pytorch_model.bin
        """,
    )

    parser.add_argument(
        "--sampler-state-dict-path",
        type=str,
        default=None,
        help="""The path to the sampler state dict if it is not None. Training will start from this sampler state dict.
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Total optimizer steps to train (the only stop condition). Must "
        "match ds_config total_num_steps so the cosine LR anneals fully.",
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use fp16 for the autocast context.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=True,
        help="Whether to use bf16 for the autocast context (recommended).",
    )

    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--save-every-n", type=int, default=2000)
    parser.add_argument("--valid-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Peak AdamW learning rate."
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=3e-5,
        help="Learning rate for SPEAR when --unfreeze-encoder is True.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clipping", type=float, default=5.0)
    parser.add_argument(
        "--ds-bucket-size",
        type=int,
        default=200000000,
        help="ZeRO communication bucket size in elements.",
    )
    parser = deepspeed.add_config_arguments(parser)
    parser.set_defaults(deepspeed=True)
    add_model_arguments(parser)

    return parser


def get_deepspeed_config(params):
    """The CE recipe's optimizer/scheduler, or an explicitly supplied DS config."""
    if params.deepspeed_config:
        return json.loads(Path(params.deepspeed_config).read_text())
    if params.max_steps <= 0 or params.gradient_accumulation_steps <= 0:
        raise ValueError("max-steps and gradient-accumulation-steps must be positive")
    if params.lr <= 0 or params.encoder_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0 < params.warmup_steps < params.max_steps:
        raise ValueError("warmup-steps must be positive and less than max-steps")
    if params.use_bf16 and params.use_fp16:
        raise ValueError("Choose BF16 or FP16, not both")
    return {
        "bf16": {"enabled": params.use_bf16},
        "fp16": {"enabled": params.use_fp16},
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": params.ds_bucket_size,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": params.ds_bucket_size,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": params.lr,
                "weight_decay": params.weight_decay,
                "betas": [0.9, 0.95],
                "eps": 1e-6,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0.0,
                "warmup_num_steps": params.warmup_steps,
                "total_num_steps": params.max_steps,
                "cos_min_ratio": 0.05,
                "warmup_type": "linear",
            },
        },
        "gradient_accumulation_steps": params.gradient_accumulation_steps,
        "gradient_clipping": params.gradient_clipping,
        "steps_per_print": params.log_interval,
        # Dynamic batches are limited by audio duration, not utterance count.
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }


def get_params() -> AttributeDict:
    """Return a dict containing training parameters."""
    params = AttributeDict(
        {
            "allowed_excess_duration_ratio": 0.1,
            "frame_shift_ms": 10,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_step": -1,
            "best_valid_step": -1,
            # Optimizer-update counter (mirrors DeepSpeed's model.global_steps),
            # NOT micro-batches: with gradient accumulation N micro-batches make
            # one global_step. All cadence (log/valid/save/stop) keys off this.
            "global_step": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 5000,
            "save_interval": 2000,
            "ce_recipe_version": CE_RECIPE_VERSION,
            "env_info": get_env_info(),
        }
    )

    return params


@dataclass
class PreparedText:
    input_ids: Tensor
    attention_mask: Tensor
    target_ids: Tensor
    num_tokens: int


def _tokenize_messages(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Preprocesses the data for supervised fine-tuning."""
    texts = []
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    for i, msg in enumerate(messages):
        ids = tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            chat_template=TEMPLATE,
            add_generation_prompt=False,
        )
        # Some transformers versions return a BatchEncoding (a UserDict, not
        # a dict) instead of a list; normalize to List[int].
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        texts.append(ids)
    # padding texts to the same length, texts is a list of list, padding with tokenizer.pad_token_id
    max_len_texts = max([len(text) for text in texts])
    if tokenizer.padding_side == "right":
        texts = [
            text + [tokenizer.pad_token_id] * (max_len_texts - len(text))
            for text in texts
        ]
    else:
        texts = [
            [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
            for text in texts
        ]
    input_ids = torch.tensor(texts, dtype=torch.int)

    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    # Match the complete role header, once per row. "assistant" is also an
    # ordinary transcript token, so searching for it alone can mask real labels.
    # The first assistant header is the one in this recipe's user/assistant pair.
    assistant_header = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)[
        "input_ids"
    ]
    for row, ids in enumerate(texts):
        prompt_end = next(
            (
                col + len(assistant_header)
                for col in range(len(ids) - len(assistant_header) + 1)
                if ids[col : col + len(assistant_header)] == assistant_header
            ),
            None,
        )
        if prompt_end is None:
            raise ValueError("the CE input is missing its assistant role header")
        target_ids[row, :prompt_end] = IGNORE_TOKEN_ID

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return input_ids, attention_mask, target_ids


def prepare_text_inputs(tokenizer: AutoTokenizer, texts: Sequence[str]) -> PreparedText:
    """Tokenize a microbatch on CPU before its accumulation window runs."""
    if not texts:
        raise ValueError("Cannot prepare an empty text batch")
    messages = [
        [
            {
                "role": "user",
                "content": f"{DEFAULT_SPEECH_TOKEN}Transcribe the speech to text.",
            },
            {"role": "assistant", "content": text.lower()},
        ]
        for text in texts
    ]
    input_ids, attention_mask, target_ids = _tokenize_messages(messages, tokenizer)
    input_ids = input_ids.to(torch.long)
    target_ids = target_ids.to(torch.long)
    num_tokens = int(target_ids.ne(IGNORE_TOKEN_ID).sum())
    if num_tokens == 0:
        raise ValueError("A CE microbatch has no supervised tokens")
    return PreparedText(input_ids, attention_mask, target_ids, num_tokens)


def accumulation_window_stats(model, num_microbatches: int, local_tokens: int):
    """Agree on window length and count supervised tokens across data-parallel ranks."""
    group = getattr(model, "data_parallel_group", None)
    world = (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if world > 1:
        device = next(model.parameters()).device
        local = torch.tensor(
            [num_microbatches, local_tokens], dtype=torch.int64, device=device
        )
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local, group=group)
        counts = torch.stack(gathered).cpu().tolist()
        sizes = [row[0] for row in counts]
        if len(set(sizes)) != 1:
            raise RuntimeError(
                f"Data-parallel ranks produced different accumulation window sizes: {sizes}"
            )
        num_microbatches = sizes[0]
        global_tokens = sum(row[1] for row in counts)
    else:
        global_tokens = local_tokens
    if num_microbatches and global_tokens <= 0:
        raise ValueError("The accumulation window has no supervised tokens")
    return num_microbatches, global_tokens, world


def reduce_step_metrics(metrics: MetricsTracker, model) -> MetricsTracker:
    """Report the same global token average used by the backward objective."""
    group = getattr(model, "data_parallel_group", None)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        keys = ("loss", "frames", "acc")
        values = torch.tensor(
            [metrics[key] for key in keys],
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
        for key, value in zip(keys, values.cpu().tolist()):
            metrics[key] = value
    return metrics


def compute_loss(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    batch: dict,
    is_training: bool,
    prepared_text: Optional[PreparedText] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """Return a microbatch token-mean loss and unnormalized reporting totals."""
    if prepared_text is None:
        prepared_text = prepare_text_inputs(tokenizer, batch["supervisions"]["text"])
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3, feature.shape
    feature = feature.to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    with torch.set_grad_enabled(is_training):
        model_outputs, acc = model(
            feature=feature,
            feature_lens=feature_lens,
            input_ids=prepared_text.input_ids.to(device),
            attention_mask=prepared_text.attention_mask.to(device),
            labels=prepared_text.target_ids.to(device),
        )
        loss = model_outputs.loss
    assert loss.requires_grad == is_training
    info = MetricsTracker()
    # "frames" is the existing MetricsTracker denominator; here it counts tokens.
    info["frames"] = prepared_text.num_tokens
    info["loss"] = loss.detach().cpu().item() * prepared_text.num_tokens
    info["acc"] = float(acc) * prepared_text.num_tokens
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    autocast_dtype = torch.bfloat16 if params.use_bf16 else torch.float16
    for batch_idx, batch in enumerate(valid_dl):
        with torch.cuda.amp.autocast(
            enabled=params.use_bf16 or params.use_fp16, dtype=autocast_dtype
        ):
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
        params.best_valid_step = params.global_step
        params.best_valid_loss = loss_value

    return tot_loss


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    rank: int = 0,
) -> None:
    from resume_ce import save

    save(params, model, train_dl, rank)


def train_one_epoch(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Normalize each optimizer update over all tokens in its complete GA window.

    Only CPU batches and tokenized text are buffered. Each microbatch's graph is
    released by its backward. A final short window is updated with its actual
    token count, and checkpoints are saved only after the window is consumed.
    """
    model.train()
    if not params.unfreeze_encoder:
        model.encoder.eval()
    if not params.unfreeze_llm:
        model.llm.eval()

    tot_loss = MetricsTracker()
    autocast_dtype = torch.bfloat16 if params.use_bf16 else torch.float16
    ga = model.gradient_accumulation_steps()
    if ga < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    batches = iter(train_dl)
    try:
        while model.global_steps < (params.stop_after_steps or params.max_steps):
            window = list(islice(batches, ga))
            prepared = [
                prepare_text_inputs(tokenizer, batch["supervisions"]["text"])
                for batch in window
            ]
            window_size, global_tokens, dp_world = accumulation_window_stats(
                model, len(window), sum(text.num_tokens for text in prepared)
            )
            if window_size == 0:
                break
            step_metrics = MetricsTracker()
            step_utts = 0
            before_step = model.global_steps
            for index, (batch, text) in enumerate(zip(window, prepared)):
                last = index == window_size - 1
                # An explicit boundary also flushes a short final window. Set it
                # before forward/backward so ZeRO's reduction hooks see it.
                model.set_gradient_accumulation_boundary(last)
                step_utts += len(batch["supervisions"]["text"])
                try:
                    with torch.cuda.amp.autocast(
                        enabled=params.use_bf16 or params.use_fp16, dtype=autocast_dtype
                    ):
                        loss, loss_info = compute_loss(
                            params=params,
                            tokenizer=tokenizer,
                            model=model,
                            batch=batch,
                            is_training=True,
                            prepared_text=text,
                        )
                    step_metrics = step_metrics + loss_info
                    # DeepSpeed divides by configured GA and averages DP ranks.
                    # Cancel both factors, then normalize the local token sum
                    # by the entire window's global token count. GA stays here
                    # even when the final window contains fewer microbatches.
                    backward_loss = loss * (
                        ga * dp_world * text.num_tokens / global_tokens
                    )
                    model.backward(backward_loss)
                    model.step()
                except Exception:
                    display_and_save_batch(batch, params=params)
                    raise
                if model.global_steps != before_step + int(last):
                    raise RuntimeError(
                        "DeepSpeed did not honor the accumulation window boundary"
                    )

            step_metrics = reduce_step_metrics(step_metrics, model)
            if step_metrics["frames"] != global_tokens:
                raise RuntimeError(
                    "Prepared and reported supervised-token counts differ"
                )
            params.global_step = model.global_steps
            step = params.global_step
            reached_max = step >= (params.stop_after_steps or params.max_steps)
            # One optimizer step done: fold this group's full-batch metrics into the
            # running average, expose them for logging, then reset the accumulators.
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + step_metrics
            cur_loss = step_metrics
            batch_size = step_utts

            if step % params.valid_interval == 0:
                logging.info("Computing validation loss")
                valid_info = compute_validation_loss(
                    params=params,
                    tokenizer=tokenizer,
                    model=model,
                    valid_dl=valid_dl,
                    world_size=world_size,
                )
                model.train()
                if not params.unfreeze_encoder:
                    model.encoder.eval()
                if not params.unfreeze_llm:
                    model.llm.eval()
                logging.info(f"Step {step}, validation: {valid_info}")
                logging.info(
                    f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
                )
                if tb_writer is not None:
                    valid_info.write_summary(tb_writer, "train/valid_", step)

            # Periodic save. Skip on the final step: it saves post-step (below) to
            # capture the last update, which would otherwise duplicate this one.
            if step % params.save_interval == 0 and not reached_max:
                save_checkpoint(
                    params=params,
                    model=model,
                    train_dl=train_dl,
                    rank=rank,
                )

            if step % params.log_interval == 0:
                try:
                    cur_lrs = scheduler.get_last_lr()
                except:  # noqa
                    cur_lrs = [0.0]
                # Two groups only with --unfreeze-encoder True (0 = SPEAR at the low LR,
                # 1 = LoRA + projector); a frozen encoder leaves one flat group.
                lr_str = "/".join(f"{lr:.2e}" for lr in cur_lrs)

                logging.info(
                    f"Step {step}/{params.max_steps}, "
                    f"loss[{cur_loss}], "
                    f"tot_loss[{tot_loss}], local batch size: {batch_size}, "
                    f"lr: {lr_str}, "
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("train/learning_rate", cur_lrs[0], step)
                    if len(cur_lrs) > 1:
                        tb_writer.add_scalar(
                            "train/learning_rate_fast", cur_lrs[1], step
                        )

                    cur_loss.write_summary(tb_writer, "train/current_", step)
                    tot_loss.write_summary(tb_writer, "train/tot_", step)

            if reached_max:
                logging.info(
                    f"Stopping at optimizer step {step} (schedule budget {params.max_steps}); saving checkpoint."
                )
                save_checkpoint(
                    params=params,
                    model=model,
                    train_dl=train_dl,
                    rank=rank,
                )
                break

    finally:
        model.set_gradient_accumulation_boundary(None)

    if tot_loss["frames"] == 0:
        # A rank may receive no batches when the dataset is tiny and drop_last
        # fragments every bucket (harmless on real-scale data).
        logging.warning("No training batches this epoch; skipping loss summary.")
        return
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_step = params.global_step
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        from :func:`get_rank` in :func:`main` (the RANK environment variable under torchrun).
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    params.save_interval = params.save_every_n
    from resume_ce import restore, resume_tag

    tag = resume_tag(params.exp_dir, params.resume)

    fix_random_seed(params.seed)
    torch.cuda.set_device(get_local_rank())
    if world_size > 1 and not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info(params)

    logging.info("About to create model")

    # 1. Speech encoder: SPEAR (consumes the 128-bin fbank the dataloader emits).
    #    Loaded from a local trust_remote_code snapshot.
    speech_encoder = load_spear_encoder(params.speech_encoder_path_or_name)
    speech_encoder_dim = spear_encoder_dim(speech_encoder)
    if params.unfreeze_encoder:
        # SPEAR ships parameters that never take part in the fine-tuning forward:
        # `mask_emb` (used only by SSL masking) and a vestigial per-layer
        # `bypass_scale` (marked "TODO: remove it" upstream, unused in forward).
        # Freeze them so they receive no gradient / optimizer state.
        for name, param in speech_encoder.named_parameters():
            if name.endswith("mask_emb") or re.search(
                r"\.layers\.\d+\.bypass_scale$", name
            ):
                param.requires_grad = False
    else:
        for name, param in speech_encoder.named_parameters():
            param.requires_grad = False

    # 2. LLM: Qwen3.
    tokenizer = AutoTokenizer.from_pretrained(params.llm_path_or_name)
    if params.use_flash_attn:
        attn_implementation = "flash_attention_2"
        tokenizer.padding_side = "left"
    else:
        attn_implementation = "sdpa"
        tokenizer.padding_side = "right"
    torch_dtype = torch.bfloat16 if params.use_bf16 else torch.float16

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path_or_name,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )

    if not params.unfreeze_llm:
        for name, param in llm.named_parameters():
            param.requires_grad = False
    else:
        if params.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "gate_proj",
                    "down_proj",
                ],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
            llm = get_peft_model(llm, lora_config)
            llm.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )
    # Qwen3's embedding table is larger than the tokenizer vocab, so the extra
    # <speech> id normally fits without resizing. Resize only if it does not.
    embed_rows = llm.get_input_embeddings().weight.shape[0]
    if llm.config.default_speech_token_id >= embed_rows:
        llm.resize_token_embeddings(len(tokenizer))

    # Full-sequence training/validation never reuses the autoregressive KV cache.
    llm.config.use_cache = False
    if params.activation_checkpointing:
        # Enable after LoRA wrapping and embedding setup, before DeepSpeed.
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()

    encoder_projector = EncoderProjector(
        speech_encoder_dim, llm.config.hidden_size, params.encoder_projector_ds_rate
    )

    model = SPEECH_LLM(
        speech_encoder,
        llm,
        encoder_projector,
        padding_side=tokenizer.padding_side,
    )
    if params.use_bf16 or params.use_fp16:
        # DeepSpeed may disable the caller's autocast in its forward. SPEAR's
        # mixed-dtype operations need autocast inside the wrapped module too.
        autocast_dtype = torch.bfloat16 if params.use_bf16 else torch.float16
        model.forward = torch.autocast("cuda", dtype=autocast_dtype)(model.forward)

    if params.pretrained_model_path:
        checkpoint = torch.load(
            params.pretrained_model_path, map_location="cpu", weights_only=False
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            logging.warning(
                f"{len(missing_keys)} keys missing when loading "
                f"{params.pretrained_model_path}, e.g. {missing_keys[:5]}"
            )
        if unexpected_keys:
            logging.warning(
                f"{len(unexpected_keys)} unexpected keys when loading "
                f"{params.pretrained_model_path}, e.g. {unexpected_keys[:5]}"
            )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    num_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info(f"Number of trainable parameters: {num_trainable}")

    if torch.cuda.is_available():
        # LOCAL_RANK, not RANK: the CUDA ordinal is node-local while RANK is global, so on a
        # second node cuda:8.. does not exist. Identical on one node, where the two agree.
        device = torch.device("cuda", get_local_rank())
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    model.to(device)

    logging.info("Using DeepSpeed ZeRO")
    # When the encoder is unfrozen it has its own, smaller peak learning rate.
    spear_params, fast_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            spear_params.append(param)
        else:  # encoder_projector.* + llm LoRA adapters
            fast_params.append(param)
    assert fast_params, "projector + LLM adapters must always be trainable."
    if spear_params:
        model_parameters = [
            {"params": spear_params, "lr": params.encoder_lr},
            {"params": fast_params, "lr": params.lr},
        ]
    else:
        # Frozen encoder: single LR group; peak LR comes from ds_config.
        model_parameters = [{"params": fast_params}]

    ds_config = get_deepspeed_config(params)
    config_path = Path(params.exp_dir) / "deepspeed_config.json"
    if tag is not None and config_path.exists():
        if json.loads(config_path.read_text()) != ds_config:
            raise ValueError("DeepSpeed configuration differs from the saved run")
    if rank == 0:
        config_path.write_text(json.dumps(ds_config, indent=2) + "\n")
    # Pass one config source; DeepSpeed rejects args.deepspeed_config + config.
    params.deepspeed_config = None
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=params, model=model, model_parameters=model_parameters, config=ds_config
    )

    if tag is not None:
        restore(params, model, tag)
        params.sampler_state_dict_path = str(Path(params.exp_dir) / f"{tag}-sampler.pt")
    data_module = LibriSpeechAsrDataModule(args)

    ds_rate = params.encoder_projector_ds_rate

    def remove_short_and_long_utt(c: Cut):
        # Too long (> 30s) uses too much memory; too short (< ds_rate encoder
        # frames) yields 0 speech tokens and crashes merge.
        if c.duration > 30.0:
            return False
        # SPEAR: fbank (10 ms shift) -> Conv2dSubsampling (T-7)//2, output rate
        # ~50 Hz. int() floors a product that can land just below the integer, so
        # this can read a frame low against encoder_out_lens.
        num_fbank = int(c.duration * 16000) // 160
        num_frames = (num_fbank - 7) // 2
        return num_frames >= ds_rate

    train_cuts = data_module.train_cuts()
    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    sampler_state_dict = None
    if params.sampler_state_dict_path:
        sampler_state_dict = torch.load(
            params.sampler_state_dict_path, weights_only=False
        )
        sampler_state_dict["max_duration"] = params.max_duration

    train_dl = data_module.train_dataloaders(
        train_cuts,
        sampler_state_dict=sampler_state_dict,
        world_size=world_size,
        rank=rank,
    )

    valid_cuts = data_module.dev_cuts()
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
    valid_dl = data_module.valid_dataloaders(
        valid_cuts, world_size=world_size, rank=rank
    )

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    assert params.max_steps > 0, "step-based training requires --max-steps > 0"
    logging.info(f"start step-based training: max_steps={params.max_steps}")
    epoch = train_dl.sampler.epoch
    empty_sweeps = 0
    while params.global_step < (params.stop_after_steps or params.max_steps):
        previous_step = params.global_step
        # One epoch == one full reshuffled sweep of the dataloader. Training
        # length is bounded by max_steps, so the last epoch is usually partial.
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.global_step)

        train_one_epoch(
            params=params,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )
        empty_sweeps = empty_sweeps + 1 if params.global_step == previous_step else 0
        if empty_sweeps >= 2:
            raise RuntimeError(
                "Two sweeps made no optimizer updates: check filtering, buckets, drop-last and batch budget"
            )
        epoch += 1

    logging.info("Done!")


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
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.stop_after_steps <= args.max_steps:
        parser.error("stop-after-steps must be between 0 and max-steps")
    if min(args.save_every_n, args.valid_interval, args.log_interval) <= 0:
        parser.error("Save, validation and log intervals must be positive")

    world_size = get_world_size()
    rank = get_rank()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warnings.filterwarnings("ignore", category=FutureWarning)
    run(rank=rank, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
