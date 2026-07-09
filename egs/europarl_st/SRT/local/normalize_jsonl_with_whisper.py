#!/usr/bin/env python3

# Copyright 2025 Nanjie Li (linanjie0820@gmail.com)
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
Normalize ASR/AST JSONL.gz datasets with Whisper's text normalization.

For every supervision entry, the script normalizes:
  * `text` using the language specified by `language`
  * `custom.st_text` using the language specified by `custom.lang`

Usage example:

python normalize_jsonl_with_whisper.py \
    --input /path/to/input.jsonl.gz \
    --output /path/to/output.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer

ENGLISH_ALIASES = {"en", "eng", "english"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize text and st_text fields in JSONL.gz using Whisper normalizers."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the source .jsonl.gz file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the destination .jsonl.gz file with normalized text.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty lines from the input (they are skipped by default).",
    )
    return parser.parse_args()


def canonicalize_lang(lang: str | None) -> str | None:
    if not lang:
        return None
    stripped = lang.strip().lower().replace("_", "-")
    if "-" in stripped:
        stripped = stripped.split("-", 1)[0]
    return stripped


def choose_normalizer(
    lang: str | None,
    english_normalizer: EnglishTextNormalizer,
    default_normalizer: BasicTextNormalizer,
):
    normalized_lang = canonicalize_lang(lang)
    if normalized_lang in ENGLISH_ALIASES:
        return english_normalizer
    return default_normalizer


def normalize_text(
    text: Any,
    lang: str | None,
    english_normalizer: EnglishTextNormalizer,
    default_normalizer: BasicTextNormalizer,
) -> Tuple[Any, bool]:
    if not isinstance(text, str):
        return text, False

    text_stripped = text.strip()
    if not text_stripped:
        return text, False

    normalizer = choose_normalizer(lang, english_normalizer, default_normalizer)
    normalized_text = normalizer(text_stripped).strip()

    # Whisper normalizers collapse whitespace; ensure we don't reintroduce leading/trailing spaces.
    if normalized_text != text:
        return normalized_text, True
    return text, False


def normalize_record(
    record: Dict[str, Any],
    english_normalizer: EnglishTextNormalizer,
    default_normalizer: BasicTextNormalizer,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    stats = {"text": 0, "st_text": 0}
    supervisions = record.get("supervisions") or []

    for supervision in supervisions:
        lang = supervision.get("language")
        text = supervision.get("text")
        normalized, changed = normalize_text(
            text, lang, english_normalizer, default_normalizer
        )
        if changed:
            supervision["text"] = normalized
            stats["text"] += 1

        custom = supervision.get("custom")
        if isinstance(custom, dict) and "st_text" in custom:
            st_lang = custom.get("lang")
            st_text = custom.get("st_text")
            normalized_st, st_changed = normalize_text(
                st_text, st_lang, english_normalizer, default_normalizer
            )
            if st_changed:
                custom["st_text"] = normalized_st
                stats["st_text"] += 1

    return record, stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    english_normalizer = EnglishTextNormalizer()
    default_normalizer = BasicTextNormalizer()

    total_lines = 0
    text_updates = 0
    st_text_updates = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as reader, gzip.open(
        output_path, "wt", encoding="utf-8"
    ) as writer:
        for line in reader:
            total_lines += 1

            stripped = line.strip("\n")
            if not stripped:
                if args.keep_empty:
                    writer.write("\n")
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {total_lines}: {exc}") from exc

            record, stats = normalize_record(
                record, english_normalizer, default_normalizer
            )
            text_updates += stats["text"]
            st_text_updates += stats["st_text"]

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Processed {total_lines} lines. "
        f"Normalized text fields: {text_updates}, st_text fields: {st_text_updates}."
    )


if __name__ == "__main__":
    main()

# Example:
# python normalize_jsonl_with_whisper.py \
#   --input /path/to/input.jsonl.gz \
#   --output /path/to/output.jsonl.gz
