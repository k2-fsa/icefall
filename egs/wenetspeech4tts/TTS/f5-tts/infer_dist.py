# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025               （authors: Yuekai Zhang）
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
# Modified from https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/cli.py
""" Example Usage
split=test_zh
llm_path=f5-tts/exp_zh/checkpoint-805000
huggingface-cli download --local-dir f5-tts-small-wenetspeech4tts-basic yuekai/f5-tts-semantic-token-small-wenetspeech4tts-basic
model_path=f5-tts-small-wenetspeech4tts-basic/epoch-10-avg-5.pt
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir ./bigvgan_v2_24khz_100band_256x
vocoder=./bigvgan_v2_24khz_100band_256x
torchrun --nproc_per_node=2 \
    f5-tts/infer_dist.py \
                --output_dir $output_dir \
                --batch_size 1 \
                --num_workers 2 \
                --llm-model-name-or-path $llm_path \
                --flow-matching-model-path $model_path \
                --decoder-dim 768 --nhead 12 --num-decoder-layers 18 \
                --use-cosyvoice-semantic-token True \
                --vocoder-dir $vocoder \
                --split-name $split -top-k 50 -top-p 0.95 -temperature 0.8 \
                --tokenizer-dir Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
from pathlib import Path

import s3tokenizer
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from bigvganinference import BigVGANInference
from datasets import load_dataset
from lhotse.serialization import load_jsonl
from llm_tts import LLMTTS
from model.modules import MelSpec
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from train import (
    add_model_arguments,
    get_model,
    get_tokenizer,
    interpolate_tokens,
    load_F5_TTS_pretrained_checkpoint,
)

from icefall.checkpoint import load_checkpoint

TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"


def get_args():
    parser = argparse.ArgumentParser(description="extract speech code")
    parser.add_argument(
        "--s3-tokenizer-name",
        required=False,
        type=str,
        choices=[
            "speech_tokenizer_v1",
            "speech_tokenizer_v1_25hz",
            "speech_tokenizer_v2_25hz",
        ],
        help="model version",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="huggingface dataset split name",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="dir to save result"
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        help="batch size (per-device) for inference",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="workers for dataloader"
    )
    parser.add_argument(
        "--prefetch", type=int, default=5, help="prefetch for dataloader"
    )
    parser.add_argument(
        "--llm-model-name-or-path",
        required=True,
        type=str,
        help="model version",
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        type=str,
        help="tokenizer dir",
    )
    parser.add_argument(
        "--vocoder-dir",
        required=True,
        type=str,
        help="vocoder dir",
    )
    parser.add_argument(
        "--flow-matching-model-path",
        required=True,
        type=str,
        help="flow matching model path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top k for sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="top p for sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="temperature for sampling",
    )
    add_model_arguments(parser)
    args = parser.parse_args()
    return args


def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels


def data_collator(batch, tokenizer, mel_spectrogram):
    speech_generation_start_index = tokenizer.convert_tokens_to_ids(
        "<|SPEECH_GENERATION_START|>"
    )
    assistant_index = tokenizer.convert_tokens_to_ids("assistant")
    target_sample_rate = 24000
    hop_length = 256
    target_rms = 0.1
    input_ids_list, ref_mel_list, ref_mel_len_list = [], [], []
    for i, item in enumerate(batch):
        prompt_text, target_text, prompt_audio_codes = (
            item["prompt_text"],
            item["target_text"],
            item["prompt_audio_cosy2_tokens"],
        )
        message = [
            {
                "role": "user",
                "content": f"Convert the text to speech: {prompt_text + target_text}",
            },
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

        input_ids = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            chat_template=TEMPLATE,
        )

        prompt_audio_codes = [c + 151665 for c in prompt_audio_codes]

        idx = input_ids.index(speech_generation_start_index)
        input_ids = input_ids[:idx] + prompt_audio_codes
        input_ids_list.append(input_ids)

        # get flow matching model's prompt mel spectrogram
        ref_audio_org, ref_sr = (
            item["prompt_audio"]["array"],
            item["prompt_audio"]["sampling_rate"],
        )
        ref_audio_org = torch.from_numpy(ref_audio_org).unsqueeze(0).float()
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio_org)))
        if ref_rms < target_rms:
            ref_audio_org = ref_audio_org * target_rms / ref_rms

        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org

        # Duration in mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length
        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        ref_mel_list.append(ref_mel)
        ref_mel_len_list.append(ref_mel_len)

    max_len = max([len(input_ids) for input_ids in input_ids_list])
    input_ids_list = [
        [tokenizer.pad_token_id] * (max_len - len(input_ids)) + input_ids
        for input_ids in input_ids_list
    ]
    input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    ids = [item["id"] for item in batch]

    ref_mel_batch = padded_mel_batch(ref_mel_list)
    ref_mel_len_batch = torch.LongTensor(ref_mel_len_list)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ids": ids,
        "ref_mel_batch": ref_mel_batch,
        "ref_mel_len_batch": ref_mel_len_batch,
    }


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    print(
        "Inference on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available()
    world_size, local_rank, rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    model = LLMTTS(
        model_dir=args.llm_model_name_or_path,
        tokenizer_dir=args.tokenizer_dir,
        s3_tokenizer_name=args.s3_tokenizer_name,
        device=device,
    )

    vocoder = BigVGANInference.from_pretrained(args.vocoder_dir, use_cuda_kernel=False)
    vocoder = vocoder.eval().to(device)

    flow_matching_model = get_model(args).eval().to(device)
    _ = load_checkpoint(
        args.flow_matching_model_path,
        model=flow_matching_model,
    )

    dataset = load_dataset(
        "yuekai/seed_tts_cosy2",
        split=args.split_name,
        trust_remote_code=True,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    mel_spectrogram = MelSpec(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="bigvgan",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=lambda x: data_collator(x, model.tokenizer, mel_spectrogram),
    )

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    for batch in dataloader:
        generate_codes = model.inference_batch(
            batch["input_ids"],
            batch["attention_mask"],
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        flow_matching_input_tokens, total_mel_lens = [], []
        for i, code in enumerate(generate_codes):
            flow_matching_input_token = interpolate_tokens(code)
            total_mel_len = len(flow_matching_input_token)
            flow_matching_input_tokens.append(flow_matching_input_token)
            total_mel_lens.append(total_mel_len)
        total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)
        ref_mels, ref_mel_lens = batch["ref_mel_batch"].to(device), batch[
            "ref_mel_len_batch"
        ].to(device)

        max_len = max([len(tokens) for tokens in flow_matching_input_tokens])
        # pad tokens to the same length
        for i, tokens in enumerate(flow_matching_input_tokens):
            flow_matching_input_tokens[i] = torch.tensor(
                tokens + [-1] * (max_len - len(tokens)), dtype=torch.long
            )
        flow_matching_input_tokens = torch.stack(flow_matching_input_tokens).to(device)
        generated, _ = flow_matching_model.sample(
            cond=ref_mels,
            text=flow_matching_input_tokens,
            duration=total_mel_lens,
            lens=ref_mel_lens,
            steps=16,
            cfg_strength=2.0,
            sway_sampling_coef=-1,
            no_ref_audio=False,
            seed=0,
        )

        for i, gen in enumerate(generated):
            gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
            gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)

            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
            target_rms = 0.1
            target_sample_rate = 24_000
            # if ref_rms_list[i] < target_rms:
            #     generated_wave = generated_wave * ref_rms_list[i] / target_rms
            utt = batch["ids"][i]
            torchaudio.save(
                f"{args.output_dir}/{utt}.wav",
                generated_wave,
                target_sample_rate,
            )

        if rank == 0:
            progress_bar.update(world_size * len(batch["ids"]))

    if rank == 0:
        progress_bar.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
