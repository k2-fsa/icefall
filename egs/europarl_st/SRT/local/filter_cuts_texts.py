#!/usr/bin/env python3

# Copyright 2026 Nanjie Li (linanjie0820@gmail.com)
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
Filter CutSet manifests by removing entries whose text or st_text is empty.

Usage example:

python filter_cuts_texts.py \
  --manifest-dir ./manifests \
  --output-dir ./manifests \
  --overwrite \
  --verbose
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove cuts from Lhotse CutSet manifests when any supervision has "
            "an empty text or st_text field."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "manifests",
        help="Directory containing *_cuts.jsonl or *_cuts.jsonl.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write filtered manifests. Defaults to manifest-dir "
            "(overwriting requires --overwrite)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rewriting files in-place (output-dir == manifest-dir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, only report statistics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def iter_manifest_files(manifest_dir: Path) -> Iterable[Path]:
    for path in sorted(manifest_dir.rglob("*.jsonl*")):
        if path.is_file() and path.name.endswith("_cuts.jsonl"):
            yield path
        elif path.is_file() and path.name.endswith("_cuts.jsonl.gz"):
            yield path


def load_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore
        for line in f:
            line = line.strip()
            if line:
                yield line


def dump_lines(path: Path, lines: Iterable[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "wt"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore
        for line in lines:
            f.write(line + "\n")


def has_empty_text(cut_obj: dict) -> bool:
    supervisions = cut_obj.get("supervisions") or []
    if not supervisions:
        return True
    for supervision in supervisions:
        text = (supervision.get("text") or "").strip()
        st_text = (supervision.get("custom", {}).get("st_text") or "").strip()
        if not text or not st_text:
            return True
    return False


def process_manifest(
    src_path: Path,
    dst_path: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    total = 0
    kept = 0
    filtered_lines = []

    for line in load_lines(src_path):
        total += 1
        obj = json.loads(line)
        if has_empty_text(obj):
            continue
        kept += 1
        if not dry_run:
            filtered_lines.append(json.dumps(obj, ensure_ascii=False))

    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dump_lines(dst_path, filtered_lines)

    return total, kept


def maybe_copy(src: Path, dst: Path) -> None:
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    manifest_dir = args.manifest_dir.resolve()
    output_dir = (args.output_dir or manifest_dir).resolve()

    if output_dir == manifest_dir and not args.overwrite and not args.dry_run:
        raise ValueError(
            "Output directory matches manifest directory. "
            "Use --overwrite to allow in-place replacement, "
            "or specify --output-dir elsewhere."
        )

    logging.info("Scanning manifests under %s", manifest_dir)
    processed = 0
    removed = 0

    for src_path in iter_manifest_files(manifest_dir):
        rel = src_path.relative_to(manifest_dir)
        dst_path = output_dir / rel
        logging.info("Filtering %s -> %s", src_path, dst_path)
        total, kept = process_manifest(
            src_path=src_path,
            dst_path=dst_path,
            dry_run=args.dry_run,
        )
        processed += 1
        removed += total - kept
        logging.info(
            "Finished %s (total=%d, kept=%d, removed=%d)",
            src_path.name,
            total,
            kept,
            total - kept,
        )

    logging.info(
        "Done. files=%d, removed=%d entries. Dry-run=%s",
        processed,
        removed,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
