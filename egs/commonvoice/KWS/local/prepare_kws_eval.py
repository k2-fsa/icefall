#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build a private, reproducible phrase-spotting evaluation from Common Voice.

The output contains paths and labels, but no audio or transcripts. Keep it
outside Git and regenerate it from the official Common Voice release.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

ASR_LOCAL = Path(__file__).resolve().parents[2] / "ASR" / "local"
if str(ASR_LOCAL) not in sys.path:
    sys.path.insert(0, str(ASR_LOCAL))

from japanese_phones import text_to_phones  # noqa: E402
from japanese_text import normalize_japanese_text  # noqa: E402


def read_keywords(path: Path):
    keywords = [
        normalize_japanese_text(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keywords or len(keywords) != len(set(keywords)):
        raise ValueError("Keyword list must be nonempty and unique after normalization")
    return keywords


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(
    root: Path, split: str, keywords, output: Path, max_rate: float, keyword_phones=None
):
    source = root / f"{split}.tsv"
    clips = root / "clips"
    counts = Counter({keyword: 0 for keyword in keywords})
    total_seconds = 0.0
    negative_seconds = 0.0
    kept = 0
    excluded = 0
    speakers = set()
    cut_ids = set()
    phone_nonmatch = Counter({keyword: 0 for keyword in keywords})
    negative_phone_collision = Counter({keyword: 0 for keyword in keywords})
    with source.open(encoding="utf-8", newline="") as stream, output.open(
        "w", encoding="utf-8"
    ) as target:
        for row in csv.DictReader(stream, delimiter="\t"):
            text = normalize_japanese_text(row["sentence"])
            audio_relative = Path("clips") / row["path"]
            audio = root / audio_relative
            if not audio.is_file():
                raise FileNotFoundError(audio)
            duration = sf.info(audio).duration
            if not text or duration <= 0 or len(text) / duration > max_rate:
                excluded += 1
                continue
            cut_id = Path(row["path"]).stem
            if cut_id in cut_ids:
                raise ValueError(f"Duplicate clip ID in {source}: {cut_id}")
            cut_ids.add(cut_id)
            labels = [keyword for keyword in keywords if keyword in text]
            if keyword_phones is not None:
                sentence_phones = text_to_phones(text)
                candidates = labels if labels else keywords
                for keyword in candidates:
                    phones = keyword_phones[keyword]
                    matches = any(
                        sentence_phones[i : i + len(phones)] == phones
                        for i in range(len(sentence_phones) - len(phones) + 1)
                    )
                    if labels and not matches:
                        phone_nonmatch[keyword] += 1
                    elif not labels and matches:
                        negative_phone_collision[keyword] += 1
            target.write(
                json.dumps(
                    {
                        "id": cut_id,
                        "audio": str(audio_relative),
                        "seconds": duration,
                        "labels": labels,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            total_seconds += duration
            if not labels:
                negative_seconds += duration
            counts.update(labels)
            if row.get("client_id"):
                speakers.add(row["client_id"])
    return {
        "split": split,
        "source_tsv_sha256": sha256(source),
        "utterances": kept,
        "excluded": excluded,
        "speakers": len(speakers),
        "audio_seconds": total_seconds,
        "negative_seconds": negative_seconds,
        "positive_by_keyword": dict(counts),
        "positive_phone_nonmatch_by_keyword": dict(phone_nonmatch),
        "negative_phone_collision_by_keyword": dict(negative_phone_collision),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commonvoice-root", type=Path, required=True)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-transcript-chars-per-second", type=float, default=20.0)
    args = parser.parse_args()
    if args.max_transcript_chars_per_second <= 0:
        parser.error("--max-transcript-chars-per-second must be positive")
    keywords = read_keywords(args.keywords_file)
    keyword_phones = {keyword: text_to_phones(keyword) for keyword in keywords}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": "Common Voice Scripted Speech 25.0, Japanese",
        "keywords": keywords,
        "selection": "fixed keyword list; normalized transcript substring",
        "max_transcript_chars_per_second": args.max_transcript_chars_per_second,
        "splits": {},
    }
    for split in ("dev", "test"):
        output = args.output_dir / f"{split}.jsonl"
        result["splits"][split] = prepare_split(
            args.commonvoice_root,
            split,
            keywords,
            output,
            args.max_transcript_chars_per_second,
            keyword_phones,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for split, stats in result["splits"].items():
        print(
            f"{split}: {stats['utterances']} cuts, "
            f"{stats['negative_seconds'] / 3600:.2f} negative hours, "
            f"{sum(stats['positive_by_keyword'].values())} positive labels"
        )


if __name__ == "__main__":
    main()
