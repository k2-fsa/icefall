"""
Calculate WER with Whisper model
"""
import argparse
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

import librosa
import soundfile as sf
import torch
from num2words import num2words
from tqdm import tqdm
from transformers import pipeline

from icefall.utils import store_transcripts, write_error_stats

logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav-path", type=str, help="path of the speech directory")
    parser.add_argument("--decode-path", type=str, help="path of the speech directory")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/huggingface/whisper_medium",
        help="path of the huggingface whisper model",
    )
    parser.add_argument(
        "--transcript-path",
        type=str,
        default="data/transcript/test.tsv",
        help="path of the transcript tsv file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="decoding batch size"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="decoding device, cuda:0 or cpu"
    )
    return parser


def post_process(text: str):
    def convert_numbers(match):
        return num2words(match.group())

    text = re.sub(r"\b\d{1,2}\b", convert_numbers, text)
    text = re.sub(r"[^a-zA-Z0-9']", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text


def save_results(
    res_dir: str,
    results: List[Tuple[str, List[str], List[str]]],
):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    recog_path = os.path.join(res_dir, "recogs.txt")
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    errs_filename = os.path.join(res_dir, "errs.txt")
    with open(errs_filename, "w") as f:
        _ = write_error_stats(f, "test", results, enable_log=True)
    logging.info("Wrote detailed error stats to {}".format(errs_filename))


class SpeechEvalDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path: str, transcript_path: str):
        super().__init__()
        self.audio_name = []
        self.audio_paths = []
        self.transcripts = []
        with Path(transcript_path).open("r", encoding="utf8") as f:
            meta = [item.split("\t") for item in f.read().rstrip().split("\n")]
        for item in meta:
            self.audio_name.append(item[0])
            self.audio_paths.append(Path(wav_path, item[0] + ".wav"))
            self.transcripts.append(item[1])

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index: int):
        audio, sampling_rate = sf.read(self.audio_paths[index])
        item = {
            "array": librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000),
            "sampling_rate": 16000,
            "reference": self.transcripts[index],
            "audio_name": self.audio_name[index],
        }
        return item


def main(args):

    batch_size = args.batch_size

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_path,
        device=args.device,
        tokenizer=args.model_path,
    )

    dataset = SpeechEvalDataset(args.wav_path, args.transcript_path)

    results = []
    bar = tqdm(
        pipe(
            dataset,
            generate_kwargs={"language": "english", "task": "transcribe"},
            batch_size=batch_size,
        ),
        total=len(dataset),
    )
    for out in bar:
        results.append(
            (
                out["audio_name"][0],
                post_process(out["reference"][0].strip()).split(),
                post_process(out["text"].strip()).split(),
            )
        )
    save_results(args.decode_path, results)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
