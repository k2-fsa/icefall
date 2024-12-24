import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

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
    load_F5_TTS_pretrained_checkpoint,
)

from icefall.checkpoint import load_checkpoint


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
        default="/home/yuekaiz/seed_tts_eval/seedtts_testset/zh/meta_head.lst",
        help="The manifest file in seed_tts_eval format",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results",
        help="The output directory to save the generated wavs",
    )

    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)
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
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
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

    metainfo = get_seedtts_testset_metainfo(args.manifest_file)
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

    vocoder = BigVGANInference.from_pretrained(
        "./bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
    )
    vocoder = vocoder.eval().to(device)

    model = get_model(args).eval().to(device)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)

    if "model_state_dict" or "ema_model_state_dict" in checkpoint:
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
