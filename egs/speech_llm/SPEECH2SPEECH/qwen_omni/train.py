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
# For Chinese dataset, you can use the following command to download the Chinese fine-tuned whisper model.
huggingface-cli download --local-dir models/whisper yuekai/icefall_asr_multi-hans-zh_whisper
# Qwen Pretrained model
huggingface-cli download --local-dir models/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct

torchrun --nproc_per_node $ngpu ./qwen_omni/train.py \
    --max-duration 50 \
    --enable-musan False \
    --exp-dir $exp_dir \
    --speech-encoder-path-or-name models/whisper/v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt \
    --llm-path-or-name Qwen/Qwen2.5-0.5B-Instruct \
    --manifest-dir data/fbank \
    --deepspeed \
    --deepspeed_config ./qwen_omni/ds_config_zero1.json \
    --use-flash-attn True \
    --use-lora True --unfreeze-llm True --unfreeze-speech-projector True --enable-speech-output True
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
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
import whisper
from data_module import AsrDataModule
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from label_smoothing import LabelSmoothingLoss
from lhotse import CutSet, load_manifest
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import IGNORE_TOKEN_ID, SPEECH_LLM, EncoderProjector
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)

# from icefall.env import get_env_info
# from icefall import diagnostics
from utils import (  # filter_uneven_sized_batch,
    AttributeDict,
    MetricsTracker,
    get_local_rank,
    get_rank,
    get_world_size,
    setup_logger,
    str2bool,
)
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

DEFAULT_SPEECH_TOKEN = "<speech>"
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--remove-whisper-encoder-input-length-restriction",
        type=str2bool,
        default=True,
        help="replace whisper encoder forward method to remove input length restriction",
    )
    parser.add_argument(
        "--llm-path-or-name",
        type=str,
        default="/workspace/asr/Qwen1.5-0.5B-Chat",
        help="Path or name of the large language model.",
    )

    parser.add_argument(
        "--speech-encoder-path-or-name",
        type=str,
        default="whisper-large-v2",
        help="Path or name of the speech encoder.",
    )

    parser.add_argument(
        "--encoder-projector-ds-rate",
        type=int,
        default=8,
        help="Downsample rate for the encoder projector.",
    )
    parser.add_argument(
        "--use-flash-attn",
        type=str2bool,
        default=True,
        help="Whether to use flash attention.",
    )

    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=False,
        help="Whether to use lora to fine-tune llm.",
    )

    parser.add_argument(
        "--enable-speech-output",
        type=str2bool,
        default=False,
        help="Whether to enable speech codec output.",
    )

    parser.add_argument(
        "--speech-tokenizer-type",
        type=str,
        default="cosyvoice2",
        help="The type of the speech tokenizer. cosyvoice2: 6561, cosyvoice1: 4096",
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
        "--exp-dir",
        type=str,
        default="whisper_qwen/exp",
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
        start from this model. e.g. ./wenetspeech/ASR/whisper/exp_large_v2/epoch-4-avg-3.pt
        """,
    )

    parser.add_argument(
        "--last-stage-model-path",
        type=str,
        default=None,
        help="""The path to the last stage model if it is not None. Training will start from this model.
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
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--unfreeze-llm",
        type=str2bool,
        default=False,
        help="Whether to unfreeze llm during training.",
    )

    parser.add_argument(
        "--unfreeze-speech-projector",
        type=str2bool,
        default=False,
        help="Whether to unfreeze speech adaptor during training.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="multi_en",
        help="The name of the dataset.",
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        help="The type of loss to use.",
    )

    parser = deepspeed.add_config_arguments(parser)
    add_model_arguments(parser)

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
            "allowed_excess_duration_ratio": 0.1,
            "subsampling_factor": 2,
            "frame_shift_ms": 10,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 1000,
        }
    )

    return params


def extract_text_and_speech_token(
    batch: dict, 
    enable_speech_output: bool
) -> Tuple[List[Dict[str, str]], Optional[List[Any]]]:
    """
    Extracts messages and speech tokens from a batch based on the dataset format.
    Uses the global DEFAULT_SPEECH_TOKEN.
    """
    messages = []
    speech_tokens = None  # Initialize as None
    if enable_speech_output:
        if "answer_cosyvoice_speech_token" in batch["supervisions"]["cut"][0].custom:
            assert "speech_token" not in batch["supervisions"]["cut"][0].custom
            speech_tokens = [
                cut.custom["answer_cosyvoice_speech_token"]
                for cut in batch["supervisions"]["cut"]
            ]
        elif "speech_token" in batch["supervisions"]["cut"][0].custom:
            speech_tokens = [
                cut.custom["speech_token"] for cut in batch["supervisions"]["cut"]
            ]
        else:
            raise ValueError("Unknown speech token type")
    answers = batch["supervisions"]["text"]
    batch_size = len(answers)

    prompt_template_dict = {
        "speech_qa": f"{DEFAULT_SPEECH_TOKEN}",
        "speech_continuation": f"Continue the following text using less than 50 words:\\n\\n{DEFAULT_SPEECH_TOKEN}",
        "asr": f"Transcribe the following audio into text:\\n\\n{DEFAULT_SPEECH_TOKEN}",
    }

    for i in range(batch_size):
        # Initialize prompt_template with the original default.
        # The 'prompt_template' argument to the function seems unused if we determine it here.
        # For now, I will proceed assuming the internal logic dictates the template.
        # If the function argument `prompt_template` was meant to be the default, this logic would need adjustment.
        current_prompt_template = "speech_qa" # Default value for prompt_template for the current item
        target = answers[i]
        message_list_item = [] 
        
        custom_data = batch["supervisions"]["cut"][i].custom

        if 'round' in custom_data:
            # slam_omni format dataset
            # For 'round' type, the current interaction's user prompt will use current_prompt_template ("speech_qa")
            current_question_with_history = custom_data["question"]
            total_round = custom_data["round"]
            history_context = current_question_with_history.rsplit("<USER>:", 1)[0].strip()
            if total_round > 1:
                history_question_answer = history_context.split("USER:")
                history_question_answer = [item for item in history_question_answer if item]
                for j in range(total_round - 1):
                    question_answer = history_question_answer[j].split("ASSISTANT:")
                    message_list_item += [
                        {"role": "user", "content": question_answer[0].strip()},
                        {"role": "assistant", "content": question_answer[1].strip()},
                    ]
        elif 'continuation' in custom_data:
            # see https://huggingface.co/datasets/fixie-ai/librispeech_asr
            ASR_PROBABILITY = 0.3
            if random.random() < ASR_PROBABILITY:
                current_prompt_template = "asr"
            else:
                current_prompt_template = "speech_continuation"
                target = custom_data["continuation"]
        else:
            # single-round, speech2speech conversation data
            pass
        message_list_item += [
            {"role": "user", "content": prompt_template_dict[current_prompt_template]},
            {"role": "assistant", "content": target},
        ]
        messages.append(message_list_item)

    return messages, speech_tokens

def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""
    texts = []
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                chat_template=TEMPLATE,
                add_generation_prompt=False,
                padding="longest",  # FIX me change padding to longest
                truncation=False,
            )
        )
    if len(texts) != len(messages):
        logging.warning(f"Remove too long text, {messages} ")
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
    # mask all tokens before token_id 151646 with IGNORE_TOKEN_ID
    # first get the indices of the tokens
    mask_prompt = True
    if mask_prompt:
        default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        mask_indices = torch.where(input_ids == default_speech_token_id)
        for i in range(mask_indices[0].size(0)):
            row = mask_indices[0][i]
            col = mask_indices[1][i]
            # + 6 to  skip: 'assistant', '\n' 151665, 151645,    198, 151644,  77091, 198
            # WAR: TODO FIXME check qwen3
            target_ids[row, : col + 6] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attention_mask, target_ids

def process_batch_text_continuation(batch: dict):
    messages = []
    transcripts = batch["supervisions"]["text"]
    continuations = [
        cut.custom["continuation"] for cut in batch["supervisions"]["cut"]
    ]
    for i in range(len(transcripts)):
        message = [
            {
                "role": "user",
                "content": f"Continue the following text using less than 50 words:\n\n{transcripts[i]}{DEFAULT_SPEECH_TOKEN}",
            },
            {"role": "assistant", "content": continuations[i]},
        ]
        messages.append(message)
    return messages

def preprocess_teacher(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""
    texts = []
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                chat_template=TEMPLATE,
                add_generation_prompt=False,
                padding="longest",  # FIX me change padding to longest
                truncation=False,
            )
        )
    if len(texts) != len(messages):
        logging.warning(f"Remove too long text, {messages} ")
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
    # mask all tokens before token_id <speech> with IGNORE_TOKEN_ID
    # first get the indices of the tokens
    mask_prompt = True
    if mask_prompt:
        default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        mask_indices = torch.where(input_ids == default_speech_token_id)
        for i in range(mask_indices[0].size(0)):
            row = mask_indices[0][i]
            col = mask_indices[1][i]
            # + 2 to  skip: 'assistant', '\n'
            # WAR: TODO FIXME check qwen3
            # THIS IS THE ONLY DIFFERENCE FROM preprocess
            target_ids[row, : col + 6] = IGNORE_TOKEN_ID
            target_ids[row, col] = default_speech_token_id
    # remove default_speech_token_id from target_ids and input_ids
    batch_size = target_ids.size(0)

    target_ids = target_ids[target_ids != default_speech_token_id].view(batch_size, -1)
    input_ids = input_ids[input_ids != default_speech_token_id].view(batch_size, -1)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attention_mask, target_ids


def compute_loss(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
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
    device = next(model.parameters()).device
    feature = batch["inputs"]

    assert feature.ndim == 3
    feature = feature.to(device)
    feature = feature.transpose(1, 2)  # (N, C, T)

    # WAR: TODO FIXME merge process_batch_slam_omni and process_batch_vocalnet
    messages, answer_cosyvoice_speech_token = extract_text_and_speech_token(
        batch, params.enable_speech_output
    )

    input_ids, attention_mask, target_ids = preprocess(messages, tokenizer)

    target_ids = target_ids.type(torch.LongTensor)
    input_ids = input_ids.type(torch.LongTensor)

    with torch.set_grad_enabled(is_training):
        if not params.enable_speech_output:
            if params.loss_type == "ce":
                loss, acc = model(
                    fbank=feature,
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=target_ids.to(device),
                )
            elif params.loss_type == "kl_div":
                messages_text = process_batch_text_continuation(batch)
                (
                    teacher_input_ids,
                    teacher_attention_mask,
                    teacher_target_ids,
                ) = preprocess_teacher(messages_text, tokenizer)
                loss, acc, acc_teacher = model.forward_kl_div(
                    fbank=feature,
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=target_ids.to(device),
                    teacher_input_ids=teacher_input_ids.to(device),
                    teacher_attention_mask=teacher_attention_mask.to(device),
                    teacher_labels=teacher_target_ids.to(device),
                )
            else:
                raise ValueError(f"Unknown loss type: {params.loss_type}")
        else:
            assert params.loss_type == "ce"
            (
                text_loss,
                acc,
                codec_loss,
                codec_acc,
                codec_topk_acc,
            ) = model.forward_with_speech_output(
                fbank=feature,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=target_ids.to(device),
                speech_codec_ids=answer_cosyvoice_speech_token,
            )
            loss = text_loss + codec_loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_lens = batch["supervisions"]["num_frames"]
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["acc"] = (
        acc * info["frames"]
    )  # WAR: to avoid normalization by the number of frames
    if params.loss_type == "kl_div":
        info["acc_teacher"] = acc_teacher * info["frames"]
    if params.enable_speech_output:
        info["codec_acc"] = codec_acc * info["frames"]
        info["codec_topk_acc"] = codec_topk_acc * info["frames"]
        info["codec_loss"] = codec_loss.detach().cpu().item()
        info["text_loss"] = text_loss.detach().cpu().item()
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    tokenizer: whisper.tokenizer.Tokenizer,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        with torch.amp.autocast("cuda", enabled=params.use_fp16):
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
    # model.encoder_projector.train()
    model.train()
    model.encoder.eval()
    if not params.unfreeze_llm:
        model.llm.eval()
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        if batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            model.encoder.eval()
            if not params.unfreeze_llm:
                model.llm.eval()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )
            if batch_idx != 0:
                model.save_checkpoint(
                    save_dir=params.exp_dir,
                    tag=f"zero-checkpoint-{params.batch_idx_train}",
                    client_state={},
                    exclude_frozen_parameters=True,
                )

                if rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(
                        params.exp_dir,
                        f"{params.exp_dir}/checkpoint-{params.batch_idx_train}",
                        tag=f"zero-checkpoint-{params.batch_idx_train}",
                        exclude_frozen_parameters=True,
                    )
                    # save sampler state dict into checkpoint
                    sampler_state_dict = train_dl.sampler.state_dict()
                    torch.save(
                        sampler_state_dict,
                        f"{params.exp_dir}/checkpoint-{params.batch_idx_train}/sampler.pt",
                    )
                    os.system(
                        f"rm -rf {params.exp_dir}/zero-checkpoint-{params.batch_idx_train}"
                    )
        try:
            with torch.amp.autocast("cuda", enabled=params.use_fp16):
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

            # deepspeed's backward() is different from torch's backward()
            # in that it does not accept a loss tensor as input.
            # It computes the loss internally.
            model.backward(loss)
            model.step()

        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if batch_idx % params.log_interval == 0:
            try:
                cur_lr = scheduler.get_last_lr()[0]
            except:  # noqa
                cur_lr = 0.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

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

    if rank == 0:
        setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info(params)
    logging.info("About to create model")

    replace_whisper_encoder_forward()
    whisper_model = whisper.load_model(params.speech_encoder_path_or_name, "cpu")
    speech_encoder = whisper_model.encoder
    speech_encoder_dim = whisper_model.dims.n_audio_state
    for name, param in speech_encoder.named_parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(params.llm_path_or_name)

    if params.use_flash_attn:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.float16
        tokenizer.padding_side = "left"

    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
        tokenizer.padding_side = "right"

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
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
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

    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )

    encoder_projector = EncoderProjector(
        speech_encoder_dim, llm.config.hidden_size, params.encoder_projector_ds_rate
    )
    if not params.unfreeze_speech_projector:
        for name, param in encoder_projector.named_parameters():
            param.requires_grad = False
        encoder_projector.eval()

    if params.enable_speech_output:
        # Determine attn_implementation and torch_dtype based on use_flash_attn
        if params.use_flash_attn:
            attn_implementation = "flash_attention_2"
            torch_dtype = torch.float16  # Or torch.bfloat16 if needed/supported
        else:
            attn_implementation = "eager"
            torch_dtype = torch.float16
        if params.speech_tokenizer_type == "cosyvoice2":
            codec_vocab_size = 6561 + 4
        elif params.speech_tokenizer_type == "cosyvoice1":
            codec_vocab_size = 4096 + 4
        else:
            raise ValueError(f"Unknown speech tokenizer type: {params.speech_tokenizer_type}")

        config = Qwen2Config(
            vocab_size=codec_vocab_size,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=2048,
            max_position_embeddings=4096,
        )

        codec_lm = AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        codec_lm.resize_token_embeddings(codec_vocab_size)
        codec_lm.vocab_size = codec_vocab_size
        codec_lm.config.pad_token_id = codec_vocab_size - 1
        codec_lm.config.eos_token_id = codec_vocab_size - 2
        codec_lm.config.bos_token_id = codec_vocab_size - 3
        codec_lm.config.mask_token_id = codec_vocab_size - 4
    else:
        codec_lm = None
    if params.loss_type == "kl_div":
        teacher_llm = AutoModelForCausalLM.from_pretrained(
            params.llm_path_or_name,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        for name, param in teacher_llm.named_parameters():
            param.requires_grad = False
        teacher_llm.eval()
    else:
        teacher_llm = None

    model = SPEECH_LLM(
        speech_encoder,
        llm,
        encoder_projector,
        codec_lm,
        codec_lm_padding_side="left" if params.use_flash_attn else "right",
        teacher_llm=teacher_llm,
    )

    if params.pretrained_model_path or params.last_stage_model_path:
        if params.pretrained_model_path is None:
            checkpoint = torch.load(params.last_stage_model_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        else:
            checkpoint = torch.load(params.pretrained_model_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            # set params.batch_idx_train according to the checkpoint name
            if "checkpoint-" in params.pretrained_model_path:
                params.batch_idx_train = int(
                    params.pretrained_model_path.split("-")[-1].split("/")[0]
                )
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    logging.info("Trainable parameters (excluding model.eval modules):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"{name}: {param.shape}")

    if torch.cuda.is_available():
        device = torch.device("cuda", get_local_rank())
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    model.to(device)

    assert params.deepspeed and world_size > 1
    logging.info("Using DeepSpeed")
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=params, model=model, model_parameters=model.parameters()
    )

    data_module = AsrDataModule(args)

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 0.8 or c.duration > 20.0:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        if "speech_token" in c.custom or "answer_cosyvoice_speech_token" in c.custom:
            codec_len = (
                len(c.custom["answer_cosyvoice_speech_token"])
                if "answer_cosyvoice_speech_token" in c.custom
                else len(c.custom["speech_token"])
            )
            if codec_len > 2200:
                logging.warning(
                    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}, lenth: {codec_len}"
                )
                return False
        if "question" in c.custom:
            if len(c.custom["question"]) > 1200:
                # logging.warning(
                #     f"Exclude cut with ID {c.id} from training. question length: {len(c.custom['question'])}"
                # )
                return False
        return True

    if params.dataset == "slam_omni_belle":
        train_cuts = data_module.train_cuts_belle()
        valid_cuts = data_module.dev_cuts_belle()
    elif params.dataset == "vocalnet_ultrachat_voiceassistant":
        train_cuts = data_module.train_cuts_en_vocalnet()
        valid_cuts = data_module.valid_cuts_en_vocalnet()
    elif params.dataset == "vocalnet_ultrachat_voiceassistant_instruct_s2s":
        train_cuts = data_module.train_cuts_en_speech2speech()
        valid_cuts = data_module.valid_cuts_en_vocalnet()
    elif params.dataset == "vocalnet_ultrachat_voiceassistant_instruct_s2s_librispeech":
        train_cuts = data_module.train_cuts_en_speech2speech_librispeech()
        valid_cuts = data_module.valid_cuts_en_vocalnet()
    elif params.dataset == "ultravox_multi_en":
        train_cuts = data_module.train_cuts_ultravox()
        valid_cuts = data_module.valid_cuts_ultravox()
    elif params.dataset == "librispeech":
        train_cuts = data_module.train_cuts_librispeech()
        valid_cuts = data_module.valid_cuts_ultravox()
    elif params.dataset == "gigaspeech":
        train_cuts = data_module.train_cuts_gigaspeech()
        valid_cuts = data_module.valid_cuts_ultravox()
    else:
        raise ValueError(f"Unknown dataset: {params.dataset}")

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)

    sampler_state_dict = None
    if params.sampler_state_dict_path:
        sampler_state_dict = torch.load(params.sampler_state_dict_path)
        sampler_state_dict["max_duration"] = params.max_duration

    train_dl = data_module.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    # train_dl = data_module.valid_dataloaders(train_cuts)
    valid_dl = data_module.valid_dataloaders(valid_cuts)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    logging.info(f"start training from epoch {params.start_epoch}")
    for epoch in range(params.start_epoch, params.num_epochs + 1):

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

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

        model.save_checkpoint(
            save_dir=params.exp_dir,
            tag=f"zero-epoch-{params.cur_epoch}",
            client_state={},
            exclude_frozen_parameters=True,
        )
        if rank == 0:
            convert_zero_checkpoint_to_fp32_state_dict(
                params.exp_dir,
                f"{params.exp_dir}/epoch-{params.cur_epoch}",
                tag=f"zero-epoch-{params.cur_epoch}",
                exclude_frozen_parameters=True,
            )
            # save sampler state dict into checkpoint
            sampler_state_dict = train_dl.sampler.state_dict()
            torch.save(
                sampler_state_dict,
                f"{params.exp_dir}/epoch-{params.cur_epoch}/sampler.pt",
            )

            os.system(f"rm -rf {params.exp_dir}/zero-epoch-{params.cur_epoch}")

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
    warnings.filterwarnings("ignore", category=FutureWarning)
    run(rank=rank, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
