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
import sys
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from cosyvoice.cli.cosyvoice import CosyVoice2
from datasets import Audio, load_dataset
from decode import audio_decode_cosyvoice2
from label_smoothing import LabelSmoothingLoss
from lhotse.utils import fix_random_seed
from model import IGNORE_TOKEN_ID, SPEECH_LLM
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from train import add_model_arguments, add_training_arguments, get_model, get_params
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from utils import (  # filter_uneven_sized_batch,
    AttributeDict,
    MetricsTracker,
    get_local_rank,
    get_rank,
    get_world_size,
    setup_logger,
    str2bool,
)

# sys.path.append("/lustre/fsw/general_sa/yuekaiz/s2s/CosyVoice/third_party/Matcha-TTS")
sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")
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
        default=1,
        help="The batch size to use.",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="test_en",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="huggingface dataset split name",
    )
    parser.add_argument(
        "--token2wav-path",
        type=str,
        default="/workspace/CosyVoice-300M-SFT",
        help="The path to the token2wav model",
    )

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
    prompt_texts, prompt_speech_16k, messages, ids, target_texts = [], [], [], [], []
    for i, item in enumerate(batch):
        # speech_tokens.append(item["prompt_audio_cosy2_tokens"])
        message_list_item = []
        message_list_item += [
            {
                "role": "user",
                "content": f"Generate a speech from the following text:\n\n{item['target_text']}{DEFAULT_SPEECH_TOKEN}",
            },
            {"role": "assistant", "content": ""},
        ]
        messages.append(message_list_item)
        target_texts.append(item["target_text"])

        ids.append(item["id"])
        prompt_texts.append(item["prompt_text"])
        speech_org = item["prompt_audio"]

        speech_org = torch.tensor(speech_org["array"], dtype=torch.float32).unsqueeze(0)
        speech_org = speech_org.mean(dim=0, keepdim=True)
        prompt_speech_16k.append(speech_org)

        # resample to 16k

    return {
        "prompt_texts": prompt_texts,
        "target_texts": target_texts,
        "prompt_speech_16k": prompt_speech_16k,
        "messages": messages,
        "ids": ids,
    }


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
    params.log_dir = Path(params.exp_dir) / "log-results-wav"
    params.log_dir.mkdir(parents=True, exist_ok=True)

    fix_random_seed(params.seed)

    if rank == 0:
        setup_logger(f"{params.exp_dir}/log/log-decode-tts")
    logging.info(params)
    logging.info("About to create model")
    model, tokenizer = get_model(params)
    if torch.cuda.is_available():
        device = torch.device("cuda", get_local_rank())
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    model.to(device)

    dataset = load_dataset("yuekai/seed_tts_cosy2", split=params.split_name)
    dataset = dataset.cast_column("prompt_audio", Audio(sampling_rate=16000))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=1,
        prefetch_factor=1,
        collate_fn=data_collator,
    )
    token2wav_model = CosyVoice2(
        params.token2wav_path, load_jit=False, load_trt=False, fp16=False
    )
    for batch in data_loader:
        messages = batch["messages"]
        prompt_texts = batch["prompt_texts"]
        prompt_speech_16k = batch["prompt_speech_16k"]
        target_texts = batch["target_texts"]
        ids = batch["ids"]
        input_ids, attention_mask, _ = preprocess(messages, tokenizer)
        generated_ids, generated_speech_output = model.decode_with_speech_output(
            None, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
        )
        generated_speech_output = [
            generated_speech_output
        ]  # WAR: only support batch = 1 for now
        for cut_id, audio_tokens, prompt_text, prompt_speech, target_text in zip(
            ids, generated_speech_output, prompt_texts, prompt_speech_16k, target_texts
        ):
            speech_file_name = params.log_dir / f"{cut_id}.wav"
            # save target_text to file
            with open(params.log_dir / f"{cut_id}.txt", "w") as f:
                f.write(f"{target_text}\n")
            audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)
            if "CosyVoice2" in params.token2wav_path:
                audio_hat = audio_decode_cosyvoice2(
                    audio_tokens,
                    prompt_text,
                    prompt_speech,
                    token2wav_model,
                )
                sf.write(speech_file_name, audio_hat.squeeze(0).cpu().numpy(), 24000)

    logging.info("Done!")


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = get_world_size()
    rank = get_rank()

    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    warnings.filterwarnings("ignore", category=FutureWarning)
    run(rank=rank, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
