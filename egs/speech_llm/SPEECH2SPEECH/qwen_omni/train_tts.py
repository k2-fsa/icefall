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
from datasets import load_dataset

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from label_smoothing import LabelSmoothingLoss

from lhotse.utils import fix_random_seed
from model import IGNORE_TOKEN_ID, SPEECH_LLM
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DistributedSampler, DataLoader

from train import add_model_arguments, add_training_arguments, get_params, get_model
from utils import (  # filter_uneven_sized_batch,
    AttributeDict,
    MetricsTracker,
    get_local_rank,
    get_rank,
    get_world_size,
    setup_logger,
    str2bool,
)

DEFAULT_SPEECH_TOKEN = "<speech>"
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="The batch size to use.",
    )

    parser = deepspeed.add_config_arguments(parser)
    add_model_arguments(parser)
    add_training_arguments(parser)
    return parser

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

def data_collator(batch):
    speech_tokens, messages, durations, ids, lang, dnsmos = [], [], [], [], [], []
    for i, item in enumerate(batch):
        speech_tokens.append(item["code"])
        message_list_item = []
        message_list_item += [
            {"role": "user", "content": f"Generate a speech from the following text:\n\n{item['text']}{DEFAULT_SPEECH_TOKEN}"},
            {"role": "assistant", "content": item["text"]},
        ]
        messages.append(message_list_item)
        durations.append(item["duration"])
        ids.append(item["id"])
        lang.append(item["language"])
        dnsmos.append(item["dnsmos"])

    return {
        "speech_tokens": speech_tokens,
        "messages": messages,
        "durations": durations,
        "ids": ids,
        "lang": lang,
        "dnsmos": dnsmos,
    }

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
    messages, answer_cosyvoice_speech_token = batch["messages"], batch["speech_tokens"]
    input_ids, attention_mask, target_ids = preprocess(messages, tokenizer)
    target_ids = target_ids.type(torch.LongTensor)
    input_ids = input_ids.type(torch.LongTensor)

    with torch.set_grad_enabled(is_training):
        (
            text_loss,
            acc,
            codec_loss,
            codec_acc,
            codec_topk_acc,
        ) = model.forward_with_speech_output(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=target_ids.to(device),
            speech_codec_ids=answer_cosyvoice_speech_token,
        )
        loss = text_loss + codec_loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    info["frames"] = len(messages)
    # Note: We use reduction=sum while computing the loss.
    info["acc"] = acc * len(messages)
    info["codec_acc"] = codec_acc * len(messages)
    info["codec_topk_acc"] = codec_topk_acc * len(messages)
    info["loss"] = loss.detach().cpu().item()
    info["codec_loss"] = codec_loss.detach().cpu().item()
    info["text_loss"] = text_loss.detach().cpu().item()
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

    # FIX ME
    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
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
    model.train()
    # model.encoder.eval()
    if not params.unfreeze_llm:
        model.llm.eval()
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["durations"])
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
            # model.encoder.eval()
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
                    # sampler_state_dict = train_dl.sampler.state_dict()
                    sampler_state_dict = train_dl.state_dict()
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

    loss_value = tot_loss["loss"]
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
    params.valid_interval = 2000

    fix_random_seed(params.seed)

    if rank == 0:
        setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info(params)
    logging.info("About to create model")
    model, tokenizer = get_model(params)
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

    sampler_state_dict = None
    if params.sampler_state_dict_path:
        sampler_state_dict = torch.load(params.sampler_state_dict_path)
    # print(params.dataset)
    ds = load_dataset(params.dataset, split="train")
    # shuffle the dataset
    ds = ds.shuffle(seed=42)
    train_test_split = ds.train_test_split(test_size=1000, seed=42)
    train_dataset, eval_dataset = train_test_split["train"], train_test_split["test"]
    # train_dataset, eval_dataset = train_test_split["test"], train_test_split["test"]

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dl = StatefulDataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=data_collator
    )
    train_dl.load_state_dict(sampler_state_dict)
    valid_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    valid_dl = DataLoader(
        eval_dataset,
        batch_size=params.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=1,
        prefetch_factor=1,
        collate_fn=data_collator
    )
   
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
            # sampler_state_dict = train_dl.sampler.state_dict()
            sampler_state_dict = train_dl.state_dict()
            torch.save(
                sampler_state_dict,
                f"{params.exp_dir}/epoch-{params.cur_epoch}/sampler.pt",
            )

            os.system(f"rm -rf {params.exp_dir}/zero-epoch-{params.cur_epoch}")

    logging.info("Done!")

def main():
    parser = get_parser()
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
