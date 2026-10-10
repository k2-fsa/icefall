#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the Common Voice Japanese model's real keyword-search decoder.

This is an utterance-level evaluation, like the WenetSpeech KWS recipe. It is
not a streaming-latency or sherpa-onnx evaluation. Output stays private.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from lhotse import KaldifeatFbank, KaldifeatFbankConfig

ICEFALL_ROOT = Path(__file__).resolve().parents[4]
if str(ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ICEFALL_ROOT))
KWS_LOCAL = Path(__file__).resolve().parents[1] / "local"
if str(KWS_LOCAL) not in sys.path:
    sys.path.insert(0, str(KWS_LOCAL))
WENETSPEECH_ZIPFORMER = ICEFALL_ROOT / "egs" / "wenetspeech" / "KWS" / "zipformer"
if str(WENETSPEECH_ZIPFORMER) not in sys.path:
    sys.path.insert(0, str(WENETSPEECH_ZIPFORMER))

from japanese_phones import text_to_phones  # noqa: E402
from prepare_kws_eval import read_keywords as read_keyword_file  # noqa: E402

from egs.commonvoice.KWS.zipformer import decode as phone_decode  # noqa: E402
from icefall import ContextGraph  # noqa: E402


def get_keyword_decoder():
    path = WENETSPEECH_ZIPFORMER / "decode.py"
    spec = importlib.util.spec_from_file_location("wenetspeech_kws_eval_decode", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--commonvoice-root", type=Path, required=True)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="epoch-60.pt")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keywords-score", type=float, default=1.5)
    parser.add_argument("--keywords-threshold", type=float, default=0.35)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--num-tailing-blanks", type=int, default=1)
    parser.add_argument("--blank-penalty", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.beam_size < 1 or args.num_threads < 1 or args.limit < 0:
        parser.error("beam-size and num-threads must be positive; limit nonnegative")
    if not 0 <= args.keywords_threshold <= 1:
        parser.error("keywords-threshold must be between 0 and 1")
    if args.num_tailing_blanks < 0:
        parser.error("num-tailing-blanks must be nonnegative")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args, keyword_decoder, device):
    recipe_args = phone_decode.get_parser().parse_args(
        [
            "--checkpoint",
            args.checkpoint,
            "--exp-dir",
            str(args.exp_dir),
            "--lang-dir",
            str(args.tokens.parent),
            "--decoding-method",
            "greedy_search",
            "--blank-penalty",
            str(args.blank_penalty),
            "--causal",
            "true",
            "--chunk-size",
            "16",
            "--left-context-frames",
            "64",
            "--num-encoder-layers",
            "1,1,1,1,1,1",
            "--feedforward-dim",
            "192,192,192,192,192,192",
            "--encoder-dim",
            "128,128,128,128,128,128",
            "--encoder-unmasked-dim",
            "128,128,128,128,128,128",
            "--decoder-dim",
            "320",
            "--joiner-dim",
            "320",
        ]
    )
    params = phone_decode.decoder.get_params()
    params.update(vars(recipe_args))
    params.exp_dir = args.exp_dir
    lexicon = phone_decode.PhoneLexicon(args.tokens.parent)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1
    model = keyword_decoder.get_model(params).to(device)
    phone_decode.load_model_for_decoding(params, model, device)
    model.eval()
    return params, model, lexicon


def read_keywords(path, token_table):
    keywords = read_keyword_file(path)
    token_ids = []
    for keyword in keywords:
        phones = text_to_phones(keyword)
        missing = [phone for phone in phones if phone not in token_table.symbols]
        if not phones or missing:
            raise ValueError(
                f"Keyword {keyword!r} has empty or missing phones: {missing}"
            )
        token_ids.append([token_table[phone] for phone in phones])
    return keywords, token_ids


def read_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(audio))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.numpy()


@torch.no_grad()
def decode_audio(audio, extractor, params, model, graph, keyword_decoder, args, device):
    features = extractor.extract(samples=audio, sampling_rate=16000)
    feature = torch.from_numpy(features).unsqueeze(0).to(device)
    feature_lens = torch.tensor([len(features)], dtype=torch.int32, device=device)
    if params.causal:
        feature_lens += 30
        feature = torch.nn.functional.pad(
            feature, pad=(0, 0, 0, 30), value=keyword_decoder.LOG_EPS
        )
    x, x_lens = model.encoder_embed(feature, feature_lens)
    padding_mask = keyword_decoder.make_pad_mask(x_lens)
    encoder_out, encoder_out_lens = model.encoder(
        x.permute(1, 0, 2), x_lens, padding_mask
    )
    hits = keyword_decoder.keywords_search(
        model=model,
        encoder_out=encoder_out.permute(1, 0, 2),
        encoder_out_lens=encoder_out_lens,
        keywords_graph=graph,
        beam=args.beam_size,
        num_tailing_blanks=args.num_tailing_blanks,
        blank_penalty=args.blank_penalty,
    )
    return [
        {
            "phrase": hit.phrase,
            "start_seconds_approx": hit.timestamps[0] * 0.04,
            "end_seconds_approx": hit.timestamps[-1] * 0.04,
            "mean_ac_prob": hit.ac_prob,
        }
        for hit in hits[0]
    ]


def main():
    args = get_args()
    if not args.tokens.is_file() or not (args.exp_dir / args.checkpoint).is_file():
        raise FileNotFoundError("Missing tokens or checkpoint")
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    keyword_decoder = get_keyword_decoder()
    params, model, lexicon = load_model(args, keyword_decoder, device)
    keywords, token_ids = read_keywords(args.keywords_file, lexicon.token_table)
    graph = ContextGraph(
        context_score=args.keywords_score, ac_threshold=args.keywords_threshold
    )
    graph.build(token_ids=token_ids, phrases=keywords)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=torch.device("cpu")))
    config = {
        "manifest_sha256": sha256(args.manifest),
        "keywords_sha256": sha256(args.keywords_file),
        "checkpoint_sha256": sha256(args.exp_dir / args.checkpoint),
        "tokens_sha256": sha256(args.tokens),
        "device": str(device),
        "keywords_score": args.keywords_score,
        "keywords_threshold": args.keywords_threshold,
        "beam_size": args.beam_size,
        "num_tailing_blanks": args.num_tailing_blanks,
        "blank_penalty": args.blank_penalty,
        "limit": args.limit,
        "positive_only": args.positive_only,
        "decoder": "Icefall keyword graph, full utterance",
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output_jsonl.with_suffix(".config.json")
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume: evaluation configuration changed")
        completed = {
            json.loads(line)["id"]
            for line in args.output_jsonl.read_text(encoding="utf-8").splitlines()
        }
    else:
        if args.output_jsonl.exists() or config_path.exists():
            raise FileExistsError("Output exists; use --resume or another output path")
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        completed = set()
    started = time.monotonic()
    processed = 0
    with args.manifest.open(encoding="utf-8") as source, args.output_jsonl.open(
        "a", encoding="utf-8"
    ) as target:
        for line in source:
            row = json.loads(line)
            if args.positive_only and not row["labels"]:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if row["id"] in completed:
                continue
            audio = read_audio(args.commonvoice_root / row["audio"])
            hits = decode_audio(
                audio, extractor, params, model, graph, keyword_decoder, args, device
            )
            target.write(
                json.dumps({"id": row["id"], "hits": hits}, ensure_ascii=False) + "\n"
            )
            target.flush()
            if processed <= 10 or processed % 100 == 0:
                print(
                    f"decoded={processed} elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    print(f"complete: {processed} selected cuts in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
