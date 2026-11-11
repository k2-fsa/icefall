#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evaluate the exported model using sherpa-onnx's streaming keyword spotter.

Requires sherpa-onnx in addition to the Icefall recipe dependencies. Audio is
fed in small chunks, but this is still a read-speech benchmark, not a real
microphone or trigger-latency measurement. Output stays private.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
import torch
import torchaudio


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path):
    samples, sample_rate = sf.read(path, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != 16000:
        samples = torchaudio.functional.resample(
            torch.from_numpy(np.asarray(samples)), sample_rate, 16000
        ).numpy()
    return np.asarray(samples, dtype=np.float32)


def decode_audio(kws, audio, chunk_samples, tail_samples):
    stream = kws.create_stream()
    hits = []

    def drain():
        while kws.is_ready(stream):
            kws.decode_stream(stream)
            phrase = kws.get_result(stream)
            if phrase:
                hits.append(
                    {"phrase": phrase, "timestamps_approx": kws.timestamps(stream)}
                )
                kws.reset_stream(stream)

    for start in range(0, len(audio), chunk_samples):
        stream.accept_waveform(16000, audio[start : start + chunk_samples])
        drain()
    stream.accept_waveform(16000, np.zeros(tail_samples, dtype=np.float32))
    stream.input_finished()
    drain()
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--commonvoice-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--keywords-score", type=float, default=1.5)
    parser.add_argument("--keywords-threshold", type=float, default=0.35)
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--num-trailing-blanks", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--feed-chunk-seconds", type=float, default=0.16)
    parser.add_argument("--tail-padding-seconds", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.keywords_threshold <= 1:
        parser.error("keywords-threshold must be between 0 and 1")
    if (
        args.max_active_paths < 1
        or args.num_trailing_blanks < 0
        or args.num_threads < 1
        or args.feed_chunk_seconds <= 0
        or args.tail_padding_seconds < 0
        or args.limit < 0
    ):
        parser.error("Invalid path, thread, chunk, padding, or limit setting")
    models = {}
    for part in ("encoder", "decoder", "joiner"):
        paths = list(args.model_dir.glob(f"{part}-*.onnx"))
        if len(paths) != 1:
            raise ValueError(f"Expected one {part} ONNX file, found {len(paths)}")
        models[part] = paths[0]
    tokens = args.model_dir / "tokens.txt"
    kws = sherpa_onnx.KeywordSpotter(
        tokens=str(tokens),
        encoder=str(models["encoder"]),
        decoder=str(models["decoder"]),
        joiner=str(models["joiner"]),
        keywords_file=str(args.keywords_file),
        num_threads=args.num_threads,
        sample_rate=16000,
        feature_dim=80,
        max_active_paths=args.max_active_paths,
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
    )
    config = {
        "manifest_sha256": sha256(args.manifest),
        "tokens_sha256": sha256(tokens),
        "keywords_sha256": sha256(args.keywords_file),
        "onnx_sha256": {part: sha256(path) for part, path in models.items()},
        "sherpa_onnx_version": sherpa_onnx.__version__,
        "keywords_score": args.keywords_score,
        "keywords_threshold": args.keywords_threshold,
        "max_active_paths": args.max_active_paths,
        "num_trailing_blanks": args.num_trailing_blanks,
        "feed_chunk_seconds": args.feed_chunk_seconds,
        "tail_padding_seconds": args.tail_padding_seconds,
        "limit": args.limit,
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
    chunk_samples = round(args.feed_chunk_seconds * 16000)
    tail_samples = round(args.tail_padding_seconds * 16000)
    started = time.monotonic()
    processed = 0
    with args.manifest.open(encoding="utf-8") as source, args.output_jsonl.open(
        "a", encoding="utf-8"
    ) as target:
        for line in source:
            row = json.loads(line)
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if row["id"] in completed:
                continue
            audio = read_audio(args.commonvoice_root / row["audio"])
            hits = decode_audio(kws, audio, chunk_samples, tail_samples)
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
