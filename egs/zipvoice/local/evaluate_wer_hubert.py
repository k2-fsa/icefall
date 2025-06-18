"""
Calculate WER with Hubert models.
"""
import argparse
import os
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from jiwer import compute_measures
from tqdm import tqdm
from transformers import pipeline


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
        help="path of the local hubert model, e.g., model/huggingface/hubert-large-ls960-ft",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="test.tsv",
        help="path of the transcript tsv file, where the first column "
        "is the wav name and the last column is the transcript",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="decoding batch size"
    )
    return parser


def post_process(text: str):
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zA-Z0-9']", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def process_one(hypo, truth):
    truth = post_process(truth)
    hypo = post_process(hypo)

    measures = compute_measures(truth, hypo)
    word_num = len(truth.split(" "))
    wer = measures["wer"]
    subs = measures["substitutions"]
    dele = measures["deletions"]
    inse = measures["insertions"]
    return (truth, hypo, wer, subs, dele, inse, word_num)


class SpeechEvalDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path: str, test_list: str):
        super().__init__()
        self.wav_name = []
        self.wav_paths = []
        self.transcripts = []
        with Path(test_list).open("r", encoding="utf8") as f:
            meta = [item.split("\t") for item in f.read().rstrip().split("\n")]
        for item in meta:
            self.wav_name.append(item[0])
            self.wav_paths.append(Path(wav_path, item[0] + ".wav"))
            self.transcripts.append(item[-1])

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index: int):
        wav, sampling_rate = sf.read(self.wav_paths[index])
        item = {
            "array": librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000),
            "sampling_rate": 16000,
            "reference": self.transcripts[index],
            "wav_name": self.wav_name[index],
        }
        return item


def main(test_list, wav_path, model_path, decode_path, batch_size, device):

    if model_path is not None:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device,
            tokenizer=model_path,
        )
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="facebook/hubert-large-ls960-ft",
            device=device,
        )

    dataset = SpeechEvalDataset(wav_path, test_list)

    bar = tqdm(
        pipe(
            dataset,
            generate_kwargs={"language": "english", "task": "transcribe"},
            batch_size=batch_size,
        ),
        total=len(dataset),
    )

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
    for out in bar:
        wav_name = out["wav_name"][0]
        transcription = post_process(out["text"].strip())
        text_ref = post_process(out["reference"][0].strip())
        truth, hypo, wer, subs, dele, inse, word_num = process_one(
            transcription, text_ref
        )
        if decode_path:
            fout.write(f"{wav_name}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n")
        wers.append(float(wer))
        inses.append(float(inse))
        deles.append(float(dele))
        subses.append(float(subs))
        word_nums += word_num

    wer = round((np.sum(subses) + np.sum(deles) + np.sum(inses)) / word_nums * 100, 3)
    subs = round(np.mean(subses) * 100, 3)
    dele = round(np.mean(deles) * 100, 3)
    inse = round(np.mean(inses) * 100, 3)
    print(f"WER: {wer}%\n")
    if decode_path:
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
        args.batch_size,
        device,
    )
