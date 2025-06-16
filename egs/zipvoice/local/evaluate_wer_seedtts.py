"""
Calculate WER with Whisper-large-v3 or Paraformer models, 
following Seed-TTS https://github.com/BytedanceSpeech/seed-tts-eval
"""

import argparse
import os
import string

import numpy as np
import scipy
import soundfile as sf
import torch
import zhconv
from funasr import AutoModel
from jiwer import compute_measures
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from zhon.hanzi import punctuation


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav-path", type=str, help="path of the speech directory")
    parser.add_argument(
        "--decode-path",
        type=str,
        default=None,
        help="path of the output file of WER information",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="path of the local whisper and paraformer model, "
        "e.g., whisper: model/huggingface/whisper-large-v3/, "
        "paraformer: model/huggingface/paraformer-zh/",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="test.tsv",
        help="path of the transcript tsv file, where the first column "
        "is the wav name and the last column is the transcript",
    )
    parser.add_argument("--lang", type=str, help="decoded language, zh or en")
    return parser


def load_en_model(model_path):
    if model_path is None:
        model_path = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    return processor, model


def load_zh_model(model_path):
    if model_path is None:
        model_path = "paraformer-zh"
    model = AutoModel(model=model_path)
    return model


def process_one(hypo, truth, lang):
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    word_num = len(truth.split(" "))
    wer = measures["wer"]
    subs = measures["substitutions"]
    dele = measures["deletions"]
    inse = measures["insertions"]
    return (truth, hypo, wer, subs, dele, inse, word_num)


def main(test_list, wav_path, model_path, decode_path, lang, device):
    if lang == "en":
        processor, model = load_en_model(model_path)
        model.to(device)
    elif lang == "zh":
        model = load_zh_model(model_path)
    params = []
    for line in open(test_list).readlines():
        line = line.strip()
        items = line.split("\t")
        wav_name, text_ref = items[0], items[-1]
        file_path = os.path.join(wav_path, wav_name + ".wav")
        assert os.path.exists(file_path), f"{file_path}"

        params.append((file_path, text_ref))
    wers = []
    inses = []
    deles = []
    subses = []
    word_nums = 0
    if decode_path:
        decode_dir = os.path.dirname(decode_path)
        if not os.path.exists(decode_dir):
            os.makedirs(decode_dir)
        fout = open(decode_path, "w")
    for wav_path, text_ref in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(wav_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(
                wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
        elif lang == "zh":
            res = model.generate(input=wav_path, batch_size_s=300, disable_pbar=True)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, "zh-cn")

        truth, hypo, wer, subs, dele, inse, word_num = process_one(
            transcription, text_ref, lang
        )
        if decode_path:
            fout.write(f"{wav_path}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n")
        wers.append(float(wer))
        inses.append(float(inse))
        deles.append(float(dele))
        subses.append(float(subs))
        word_nums += word_num

    wer_avg = round(np.mean(wers) * 100, 3)
    wer = round((np.sum(subses) + np.sum(deles) + np.sum(inses)) / word_nums * 100, 3)
    subs = round(np.mean(subses) * 100, 3)
    dele = round(np.mean(deles) * 100, 3)
    inse = round(np.mean(inses) * 100, 3)
    print(f"Seed-TTS WER: {wer_avg}%\n")
    print(f"WER: {wer}%\n")
    if decode_path:
        fout.write(f"SeedTTS WER: {wer_avg}%\n")
        fout.write(f"WER: {wer}%\n")
        fout.flush()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    main(
        args.test_list,
        args.wav_path,
        args.model_path,
        args.decode_path,
        args.lang,
        device,
    )
