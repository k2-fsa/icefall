#!/usr/bin/env python3
# Modified from https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/eval/eval_infer_batch.py
"""
Usage:
# docker: ghcr.io/swivid/f5-tts:main
# pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install kaldialign lhotse tensorboard bigvganinference sentencepiece sherpa-onnx
# huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir bigvgan_v2_24khz_100band_256x
manifest=/path/seed_tts_eval/seedtts_testset/zh/meta.lst
python3 f5-tts/generate_averaged_model.py \
    --epoch 56 \
    --avg 14 --decoder-dim 768 --nhead 12 --num-decoder-layers 18 \
    --exp-dir exp/f5_small

# command for text token input
accelerate launch f5-tts/infer.py --nfe 16 --model-path $model_path --manifest-file $manifest --output-dir $output_dir --decoder-dim 768 --nhead 12 --num-decoder-layers 18

# command for cosyvoice semantic token input
split=test_zh # seed_tts_eval test_zh
accelerate launch f5-tts/infer.py --nfe 16 --model-path $model_path --split-name $split --output-dir $output_dir --decoder-dim 768 --nhead 12 --num-decoder-layers 18 --use-cosyvoice-semantic-token True

bash local/compute_wer.sh $output_dir $manifest
"""
import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from bigvganinference import BigVGANInference
from model.cfm import CFM
from model.dit import DiT
from model.modules import MelSpec
from model.utils import convert_char_to_pinyin
from tqdm import tqdm
from train import (
    add_model_arguments,
    get_model,
    get_tokenizer,
    interpolate_tokens,
    load_F5_TTS_pretrained_checkpoint,
)

from icefall.checkpoint import load_checkpoint
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="f5-tts/vocab.txt",
        help="Path to the unique text tokens file",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/yuekaiz/HF/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt",
        help="Path to the unique text tokens file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--nfe",
        type=int,
        default=16,
        help="The number of steps for the neural ODE",
    )

    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="The manifest file in seed_tts_eval format",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results",
        help="The output directory to save the generated wavs",
    )

    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument(
        "--interpolate-token",
        type=str2bool,
        default=True,
        help="Interpolate semantic token to match mel frames for CosyVoice",
    )

    parser.add_argument(
        "--use-cosyvoice-semantic-token",
        type=str2bool,
        default=False,
        help="Whether to use cosyvoice semantic token to replace text token.",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="huggingface dataset split name",
    )

    add_model_arguments(parser)
    return parser.parse_args()


def get_inference_prompt(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="bigvgan",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(
        metainfo, desc="Processing prompts..."
    ):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert (
            ref_audio.shape[-1] > 5000
        ), f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [prompt_text + gt_text]
        if tokenizer == "pinyin":
            text_list = convert_char_to_pinyin(text, polyphone=polyphone)
        else:
            text_list = text

        # Duration, mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length
        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len = ref_mel_len + int(
                ref_mel_len / ref_text_len * gen_text_len / speed
            )

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor(
            (total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets
        )

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


def get_inference_prompt_cosy_voice_huggingface(
    dataset,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="bigvgan",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
    interpolate_token=False,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for i in range(len(dataset)):
        utt = dataset[i]["id"]
        ref_audio_org, ref_sr = (
            dataset[i]["prompt_audio"]["array"],
            dataset[i]["prompt_audio"]["sampling_rate"],
        )
        ref_audio_org = torch.from_numpy(ref_audio_org).unsqueeze(0).float()
        audio_tokens = dataset[i]["target_audio_cosy2_tokens"]
        prompt_audio_tokens = dataset[i]["prompt_audio_cosy2_tokens"]

        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio_org)))
        if ref_rms < target_rms:
            ref_audio_org = ref_audio_org * target_rms / ref_rms

        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org
        input_tokens = prompt_audio_tokens + audio_tokens

        if interpolate_token:
            input_tokens = interpolate_tokens(input_tokens)
        text_list = input_tokens

        # Duration, mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length

        total_mel_len = len(input_tokens)
        if not interpolate_token:
            total_mel_len = int(total_mel_len / 4 * 15)

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        if total_mel_len > max_tokens:
            print(
                f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
            )
            continue
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor(
            (total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets
        )

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        # final_text_list[bucket_i].extend(text_list)
        final_text_list[bucket_i].append(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


def inference_speech_token(
    cosyvoice,
    tts_text,
    prompt_text,
    prompt_speech_16k,
    stream=False,
    speed=1.0,
    text_frontend=True,
):
    tokens = []
    prompt_text = cosyvoice.frontend.text_normalize(
        prompt_text, split=False, text_frontend=text_frontend
    )
    for i in cosyvoice.frontend.text_normalize(
        tts_text, split=True, text_frontend=text_frontend
    ):

        tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token(i)
        (
            prompt_text_token,
            prompt_text_token_len,
        ) = cosyvoice.frontend._extract_text_token(prompt_text)
        speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(
            prompt_speech_16k
        )

        for i in cosyvoice.model.llm.inference(
            text=tts_text_token.to(cosyvoice.model.device),
            text_len=torch.tensor([tts_text_token.shape[1]], dtype=torch.int32).to(
                cosyvoice.model.device
            ),
            prompt_text=prompt_text_token.to(cosyvoice.model.device),
            prompt_text_len=torch.tensor(
                [prompt_text_token.shape[1]], dtype=torch.int32
            ).to(cosyvoice.model.device),
            prompt_speech_token=speech_token.to(cosyvoice.model.device),
            prompt_speech_token_len=torch.tensor(
                [speech_token.shape[1]], dtype=torch.int32
            ).to(cosyvoice.model.device),
            embedding=None,
        ):
            tokens.append(i)
    return tokens, speech_token


def get_inference_prompt_cosy_voice(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="bigvgan",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
    interpolate_token=False,
):

    import sys

    # please change the path to the cosyvoice accordingly
    sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")
    sys.path.append("/workspace/CosyVoice")
    from cosyvoice.cli.cosyvoice import CosyVoice2

    # please download the cosyvoice model first
    cosyvoice = CosyVoice2(
        "/workspace/CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=False
    )

    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(
        metainfo, desc="Processing prompts..."
    ):
        # Audio
        ref_audio_org, ref_sr = torchaudio.load(prompt_wav)

        # cosy voice
        if ref_sr != 16000:
            resampler = torchaudio.transforms.Resample(ref_sr, 16000)
            ref_audio_16k = resampler(ref_audio_org)
        else:
            ref_audio_16k = ref_audio_org
        audio_tokens, prompt_audio_tokens = inference_speech_token(
            cosyvoice, gt_text, prompt_text, ref_audio_16k, stream=False
        )

        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio_org)))
        if ref_rms < target_rms:
            ref_audio_org = ref_audio_org * target_rms / ref_rms
        assert (
            ref_audio_org.shape[-1] > 5000
        ), f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org

        # Text
        # if len(prompt_text[-1].encode("utf-8")) == 1:
        #     prompt_text = prompt_text + " "
        # text = [prompt_text + gt_text]
        # if tokenizer == "pinyin":
        #     text_list = convert_char_to_pinyin(text, polyphone=polyphone)
        # else:
        #     text_list = text

        # concat two tensors: prompt audio tokens with audio tokens --> shape 1, prompt_audio_tokens + audio_tokens
        # prompt_audio_tokens shape 1, prompt_audio_tokens
        # audio_tokens shape 1, audio_tokens
        prompt_audio_tokens = prompt_audio_tokens.squeeze().cpu().tolist()
        input_tokens = prompt_audio_tokens + audio_tokens

        # convert it into a list
        # input_tokens_list = input_tokens.squeeze().cpu().tolist()
        if interpolate_token:
            input_tokens = interpolate_tokens(input_tokens)
        text_list = input_tokens

        # Duration, mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length
        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len_compute = ref_mel_len + int(
                ref_mel_len / ref_text_len * gen_text_len / speed
            )
            total_mel_len = len(input_tokens)
            if not interpolate_token:
                total_mel_len = int(total_mel_len / 4 * 15)
            print(
                f"total_mel_len_compute: {total_mel_len_compute}, total_mel_len: {total_mel_len}"
            )

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor(
            (total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets
        )

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        # final_text_list[bucket_i].extend(text_list)
        final_text_list[bucket_i].append(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels


def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        assert len(line.strip().split("|")) == 4
        utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
        utt = Path(utt).stem
        gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


def main():
    args = get_parser()

    accelerator = Accelerator()
    device = f"cuda:{accelerator.process_index}"
    if args.manifest_file:
        metainfo = get_seedtts_testset_metainfo(args.manifest_file)
        if not args.use_cosyvoice_semantic_token:
            prompts_all = get_inference_prompt(
                metainfo,
                speed=1.0,
                tokenizer="pinyin",
                target_sample_rate=24_000,
                n_mel_channels=100,
                hop_length=256,
                mel_spec_type="bigvgan",
                target_rms=0.1,
                use_truth_duration=False,
                infer_batch_size=1,
            )
        else:
            prompts_all = get_inference_prompt_cosy_voice(
                metainfo,
                speed=1.0,
                tokenizer="pinyin",
                target_sample_rate=24_000,
                n_mel_channels=100,
                hop_length=256,
                mel_spec_type="bigvgan",
                target_rms=0.1,
                use_truth_duration=False,
                infer_batch_size=1,
                interpolate_token=args.interpolate_token,
            )
    else:
        assert args.use_cosyvoice_semantic_token
        dataset = datasets.load_dataset(
            "yuekai/seed_tts_cosy2",
            split=args.split_name,
            trust_remote_code=True,
        )
        prompts_all = get_inference_prompt_cosy_voice_huggingface(
            dataset,
            speed=1.0,
            tokenizer="pinyin",
            target_sample_rate=24_000,
            n_mel_channels=100,
            hop_length=256,
            mel_spec_type="bigvgan",
            target_rms=0.1,
            use_truth_duration=False,
            infer_batch_size=1,
            interpolate_token=args.interpolate_token,
        )

    vocoder = BigVGANInference.from_pretrained(
        "./bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
    )
    vocoder = vocoder.eval().to(device)

    model = get_model(args).eval().to(device)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if "ema_model_state_dict" in checkpoint or "model_state_dict" in checkpoint:
        model = load_F5_TTS_pretrained_checkpoint(model, args.model_path)
    else:
        _ = load_checkpoint(
            args.model_path,
            model=model,
        )

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            (
                utts,
                ref_rms_list,
                ref_mels,
                ref_mel_lens,
                total_mel_lens,
                final_text_list,
            ) = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            if args.use_cosyvoice_semantic_token:
                # concat final_text_list
                max_len = max([len(tokens) for tokens in final_text_list])
                # pad tokens to the same length
                for i, tokens in enumerate(final_text_list):
                    final_text_list[i] = torch.tensor(
                        tokens + [-1] * (max_len - len(tokens)), dtype=torch.long
                    )
                final_text_list = torch.stack(final_text_list).to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=args.nfe,
                    cfg_strength=2.0,
                    sway_sampling_coef=args.swaysampling,
                    no_ref_audio=False,
                    seed=args.seed,
                )
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)

                    generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
                    target_rms = 0.1
                    target_sample_rate = 24_000
                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(
                        f"{args.output_dir}/{utts[i]}.wav",
                        generated_wave,
                        target_sample_rate,
                    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60 :.2f} minutes.")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
