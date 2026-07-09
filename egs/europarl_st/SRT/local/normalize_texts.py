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

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

try:
    from whisper.normalizers import (
        BasicTextNormalizer as WhisperBasicTextNormalizer,  # type: ignore
    )
    from whisper.normalizers import (
        EnglishTextNormalizer as WhisperEnglishTextNormalizer,
    )
except Exception:  # pragma: no cover - only triggered when whisper isn't installed
    WhisperBasicTextNormalizer = None
    WhisperEnglishTextNormalizer = None


class LocalBasicTextNormalizer:
    """Language-agnostic text cleaner inspired by Whisper's BasicTextNormalizer."""

    _WS_RE = re.compile(r"\s+")
    _REPLACEMENTS = str.maketrans(
        {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2015": "-",
            "\u2212": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201A": "'",
            "\u201B": "'",
            "\u2032": "'",
            "\u2035": "'",
            "\u201C": '"',
            "\u201D": '"',
            "\u201E": '"',
            "\u00AB": '"',
            "\u00BB": '"',
            "\u02BC": "'",
            "\u0060": "'",
            "\u00B4": "'",
            "\u200B": " ",
            "\u200C": " ",
            "\u200D": " ",
            "\u200E": " ",
            "\u200F": " ",
            "\u202A": " ",
            "\u202B": " ",
            "\u202C": " ",
            "\u202D": " ",
            "\u202E": " ",
            "\u2060": " ",
            "\ufeff": " ",
            "\u00A0": " ",
        }
    )

    def __call__(self, text: str) -> str:
        if text is None:
            return ""
        normalized = unicodedata.normalize("NFKC", str(text))
        normalized = normalized.translate(self._REPLACEMENTS)
        normalized = self._strip_control_characters(normalized)
        normalized = self._WS_RE.sub(" ", normalized).strip()
        return normalized

    @staticmethod
    def _strip_control_characters(text: str) -> str:
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("C"):
                # Preserve standard whitespace while collapsing it later.
                cleaned.append(" " if ch.isspace() else "")
            else:
                cleaned.append(ch)
        return "".join(cleaned)


class LocalEnglishTextNormalizer(LocalBasicTextNormalizer):
    """Very small subset of Whisper's EnglishTextNormalizer for offline use."""

    _ASCII_ONLY_RE = re.compile(r"[^a-z0-9' ]+")

    def __call__(self, text: str) -> str:
        normalized = super().__call__(text)
        normalized = normalized.lower()
        normalized = self._ASCII_ONLY_RE.sub(" ", normalized)
        normalized = self._WS_RE.sub(" ", normalized).strip()
        return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Whisper-style text normalization to JSONL files and write "
            "the results to a 'normalizer' subdirectory."
        )
    )
    default_src = Path(__file__).resolve().parent.parent / "texts"
    default_dst = default_src / "normalizer"
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=default_src,
        help="Root directory containing JSONL files (default: %(default)s).",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=default_dst,
        help="Destination root for normalized JSONL files (default: %(default)s).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=("text", "st_text"),
        help=(
            "JSON keys to normalize (default: text st_text). "
            "Supports dot-separated paths."
        ),
    )
    parser.add_argument(
        "--normalizer",
        choices=("basic", "english"),
        default="basic",
        help="Which normalization preset to use (default: basic).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose normalized counterpart already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only report statistics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-auto-english",
        dest="auto_english",
        action="store_false",
        help=(
            "Disable automatic detection of English fields based on "
            "language-pair directory names."
        ),
    )
    parser.set_defaults(auto_english=True)
    return parser.parse_args()


def load_normalizer(name: str):
    if name == "english":
        if WhisperEnglishTextNormalizer is not None:
            logging.info("Using whisper.normalizers.EnglishTextNormalizer")
            return WhisperEnglishTextNormalizer()
        logging.info("Using built-in LocalEnglishTextNormalizer")
        return LocalEnglishTextNormalizer()
    if WhisperBasicTextNormalizer is not None:
        logging.info("Using whisper.normalizers.BasicTextNormalizer")
        return WhisperBasicTextNormalizer()
    logging.info("Using built-in LocalBasicTextNormalizer")
    return LocalBasicTextNormalizer()


def iter_jsonl_files(src_dir: Path) -> Iterator[Path]:
    for jsonl_path in sorted(src_dir.rglob("*.jsonl")):
        if jsonl_path.is_file():
            yield jsonl_path


def ensure_destination_path(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    rel_path = src_file.relative_to(src_root)
    dst_file = dst_root / rel_path
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    return dst_file


def set_nested_field(obj: dict, dotted_key: str, value: str) -> bool:
    parts = dotted_key.split(".")
    current = obj
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False

    last_key = parts[-1]
    if isinstance(current, dict) and last_key in current:
        current[last_key] = value
        return True
    return False


def get_nested_field(obj: dict, dotted_key: str):
    current = obj
    for part in dotted_key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def normalize_line(
    obj: dict,
    fields: Sequence[str],
    default_normalizer: Callable[[str], str],
    field_normalizers: Dict[str, Callable[[str], str]],
) -> int:
    changed = 0
    for field in fields:
        text_value = get_nested_field(obj, field)
        if not isinstance(text_value, str):
            continue
        normalize = field_normalizers.get(field, default_normalizer)
        normalized = normalize(text_value).strip()
        if normalized != text_value:
            set_nested_field(obj, field, normalized)
            changed += 1
    return changed


def process_file(
    src_path: Path,
    dst_path: Path,
    fields: Sequence[str],
    default_normalizer: Callable[[str], str],
    field_normalizers: Dict[str, Callable[[str], str]],
    dry_run: bool = False,
) -> tuple[int, int]:
    total = 0
    changed_lines = 0
    newline = "\n"
    with src_path.open("r", encoding="utf-8") as src, (
        open(dst_path, "w", encoding="utf-8", newline=newline)
        if not dry_run
        else nullcontext()
    ) as dst:  # type: ignore
        for raw_line in src:
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                continue
            total += 1
            obj = json.loads(raw_line)
            changed = normalize_line(
                obj,
                fields,
                default_normalizer,
                field_normalizers,
            )
            if changed:
                changed_lines += 1
            if not dry_run:
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return total, changed_lines


class nullcontext:
    """Minimal stand-in for contextlib.nullcontext for Python < 3.7 environments."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def looks_like_lang_pair(name: str) -> bool:
    return "_" in name and "." not in name


def parse_lang_pair(name: str) -> Optional[Tuple[str, str]]:
    if not looks_like_lang_pair(name):
        return None
    src, dst = name.split("_", 1)
    if not src or not dst:
        return None
    return (src, dst)


def detect_lang_pair(
    src_dir: Path, src_file: Path
) -> Tuple[Optional[str], Optional[str]]:
    parsed = parse_lang_pair(src_dir.name)
    if parsed:
        return parsed
    try:
        relative = src_file.relative_to(src_dir)
    except ValueError:
        return (None, None)
    if not relative.parts:
        return (None, None)
    parsed = parse_lang_pair(relative.parts[0])
    if parsed:
        return parsed
    return (None, None)


def choose_field_normalizers(
    src_dir: Path,
    src_file: Path,
    fields: Sequence[str],
    english_normalizer: Callable[[str], str],
    auto_english: bool,
) -> Dict[str, Callable[[str], str]]:
    field_map: Dict[str, Callable[[str], str]] = {}
    if not auto_english:
        return field_map

    src_lang, dst_lang = detect_lang_pair(src_dir, src_file)
    if not src_lang and not dst_lang:
        return field_map

    src_lang = (src_lang or "").lower()
    dst_lang = (dst_lang or "").lower()

    if src_lang.startswith("en") and "text" in fields:
        field_map["text"] = english_normalizer
    if dst_lang.startswith("en") and "st_text" in fields:
        field_map["st_text"] = english_normalizer
    return field_map


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    basic_normalizer = load_normalizer("basic")
    english_normalizer = load_normalizer("english")
    default_normalizer = (
        english_normalizer if args.normalizer == "english" else basic_normalizer
    )
    args.src_dir = args.src_dir.resolve()
    args.dst_dir = args.dst_dir.resolve()
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Normalizing JSONL files from %s into %s (fields=%s, dry_run=%s)",
        args.src_dir,
        args.dst_dir,
        ", ".join(args.fields),
        args.dry_run,
    )

    processed_files = 0
    total_lines = 0
    total_changed = 0

    for src_file in iter_jsonl_files(args.src_dir):
        dst_file = ensure_destination_path(src_file, args.src_dir, args.dst_dir)
        if args.skip_existing and dst_file.is_file():
            logging.info("Skipping %s (already exists)", src_file)
            continue

        logging.info("Processing %s -> %s", src_file, dst_file)
        field_normalizers = choose_field_normalizers(
            args.src_dir,
            src_file,
            args.fields,
            english_normalizer,
            args.auto_english,
        )
        if field_normalizers and args.verbose:
            logging.debug(
                "English normalizer applied to fields %s for %s",
                ", ".join(sorted(field_normalizers)),
                src_file,
            )
        file_total, file_changed = process_file(
            src_file,
            dst_file,
            args.fields,
            default_normalizer,
            field_normalizers,
            dry_run=args.dry_run,
        )
        processed_files += 1
        total_lines += file_total
        total_changed += file_changed
        logging.info(
            "Finished %s (lines=%d, changed=%d)",
            src_file.name,
            file_total,
            file_changed,
        )

    logging.info(
        "Done. files=%d, lines=%d, lines_changed=%d",
        processed_files,
        total_lines,
        total_changed,
    )


if __name__ == "__main__":
    main()


"""
Example usage:

python normalize_texts.py \
  --src-dir ./texts \
  --dst-dir ./normalizer \
  --fields text st_text \
  --normalizer basic \
  --skip-existing

Options:
  --src-dir / --dst-dir: Input and output directories (defaults are relative to this script).
  --fields: JSON keys to normalize (supports dot-separated paths like nested.field).
            Default: text st_text.
  --normalizer basic|english: Choose the base normalizer. If whisper.normalizers is installed,
            the official implementation is used; otherwise falls back to built-in
            LocalBasicTextNormalizer / LocalEnglishTextNormalizer.
  --auto-english (enabled by default): Automatically detects English fields from directory
            names like 'es_en' and applies the English normalizer (keeps only [a-z0-9']).
            Use --no-auto-english to disable.
  --skip-existing: Skip files that already have a corresponding output.
  --dry-run: Only report statistics without writing to disk.
  --verbose: Enable DEBUG-level logging.
"""
