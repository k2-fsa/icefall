#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
# Copyright    2024                            (authors: Yuekai Zhang)
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
This script is used to synthesize speech from text prompts and audio prompts.
Usage example:
    python3 valle/infer.py --output-dir demos_epoch_${epoch}_avg_${avg} \
        --checkpoint=${exp_dir}/epoch-${epoch}-avg-${avg}.pt \
        --text-prompts "KNOT one point one five miles per hour." \
        --audio-prompts ./prompts/8463_294825_000043_000000.wav \
        --text "To get up and running quickly just follow the steps below."

    top_p=1.0
    python3 valle/infer.py --output-dir demos_epoch_${epoch}_avg_${avg}_top_p_${top_p} \
            --top-k -1 --temperature 1.0 \
            --text ./aishell3.txt \
            --checkpoint ${exp_dir}/epoch-${epoch}-avg-${avg}.pt \
            --text-extractor pypinyin_initials_finals --top-p ${top_p}

"""
import argparse
import logging
import os
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from compute_neural_codec_and_prepare_text_tokens import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_text,
)
from encodec.utils import convert_audio
from k2 import symbol_table
from tokenizer import get_text_token_collater
from valle import VALLE

from icefall.utils import AttributeDict, str2bool


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="prompt text\t prompt audio\ttarget text\ttarget audio",
    )

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./valle/exp/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Whether AR Decoder do top_p(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--repetition-aware-sampling",
        type=str2bool,
        default=False,
        help="Whether AR Decoder do valle-2 repetition-aware sampling. https://arxiv.org/pdf/2406.05370",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)

    params = AttributeDict(checkpoint)
    model = VALLE(
        params.decoder_dim,
        params.nhead,
        params.num_decoder_layers,
        norm_first=params.norm_first,
        add_prenet=params.add_prenet,
        prefix_mode=params.prefix_mode,
        share_embedding=params.share_embedding,
        nar_scale_factor=params.scale_factor,
        prepend_bos=params.prepend_bos,
        num_quantizers=params.num_quantizers,
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    return model, params.text_tokens


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


@torch.no_grad()
def main():
    args = get_args()
    text_tokenizer = TextTokenizer(backend=args.text_extractor)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model, text_tokens = load_model(args.checkpoint, device)

    text_collater = get_text_token_collater(text_tokens)

    audio_tokenizer = AudioTokenizer()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(f"{args.output_dir}/p{n}.wav", samples[0], 24000)

            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    if os.path.isfile(args.text):  # for demos
        # https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/prepare.py
        with open(args.text) as f:
            for line in f:
                fields = line.strip().split("  ")
                fields = [item for item in fields if item]
                assert len(fields) == 4
                prompt_text, prompt_audio, text, audio_path = fields
                logging.info(f"synthesize text: {text}")
                text_tokens, text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text} {text}".strip()
                        )
                    ]
                )
                _, enroll_x_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=f"{prompt_text}".strip())]
                )

                audio_prompts = tokenize_audio(audio_tokenizer, prompt_audio)
                audio_prompts = audio_prompts[0][0].transpose(2, 1).to(device)

                # synthesis
                encoded_frames = model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    ras=args.repetition_aware_sampling,
                )

                samples = audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
                # store
                # save audio path into args.output_dir + audio_path
                audio_path = f"{args.output_dir}/{audio_path}"
                # mkdir -p
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                torchaudio.save(audio_path, samples[0].cpu(), 24000)
        return

    for n, text in enumerate(args.text.split("|")):
        logging.info(f"synthesize text: {text}")
        text_tokens, text_tokens_lens = text_collater(
            [tokenize_text(text_tokenizer, text=f"{text_prompts} {text}".strip())]
        )

        # synthesis
        enroll_x_lens = None
        if text_prompts:
            _, enroll_x_lens = text_collater(
                [tokenize_text(text_tokenizer, text=f"{text_prompts}".strip())]
            )
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=args.top_k,
            temperature=args.temperature,
            top_p=args.top_p,
            ras=args.repetition_aware_sampling,
        )

        if audio_prompts != []:
            samples = audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])
            # store
            torchaudio.save(f"{args.output_dir}/{n}.wav", samples[0].cpu(), 24000)
        else:  # Transformer
            pass


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
