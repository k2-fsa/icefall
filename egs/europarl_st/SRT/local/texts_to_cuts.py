#!/usr/bin/env python3
# Copyright 2025 Authors
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
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

try:
    import torch  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "texts_to_cuts.py now requires torch. Please install PyTorch."
    ) from exc

try:
    from lhotse import (
        CutSet,
        Features,
        KaldifeatFbank,
        KaldifeatFbankConfig,
        LilcomChunkyWriter,
        Recording,
        RecordingSet,
        SupervisionSegment,
        SupervisionSet,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "texts_to_cuts.py now depends on lhotse (pip install lhotse)."
    ) from exc


def parse_args() -> argparse.Namespace:
    default_src = Path(__file__).resolve().parent.parent / "texts" / "normalizer"
    default_dst = (default_src.parents[1] / "cut_manifests").resolve()
    default_storage = (default_dst.parents[0] / "fbank_storage").resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Convert normalized JSONL files into CutSet manifests that already "
            "contain kaldifeat FBANK features (MonoCut entries with 'features')."
        )
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=default_src,
        help="Root directory containing normalized *.jsonl files.",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=default_dst,
        help="Output directory for generated *.jsonl.gz files (mirrors src layout).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Base directory that the JSON 'source' field is relative to.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=default_storage,
        help="Directory where .lca feature chunks will be written.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers used when computing FBANK features.",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="Total audio seconds per minibatch for feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for feature extraction (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--train-shard-duration",
        type=float,
        default=0.0,
        help=(
            "For train splits only: limit total audio seconds per feature shard. "
            "Each shard writes its own .lca when > 0 (disabled by default)."
        ),
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip entries whose audio file is missing instead of raising.",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=default_storage / "feature_cache.json",
        help="JSON file used to memoize feature descriptors per recording.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore any existing cache entries and recompute features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *.jsonl.gz outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def iter_jsonl_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.jsonl")):
        if path.is_file():
            yield path


def ensure_destination_path(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    rel = src_file.relative_to(src_root)
    base = src_file.name[:-6] if src_file.name.endswith(".jsonl") else src_file.stem
    dst_name = f"{base}_cuts.jsonl.gz"
    dst_path = dst_root / rel.parent / dst_name
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return dst_path


def parse_lang_pair(path: Path) -> Tuple[str, str]:
    parent_name = path.parent.name
    if "_" not in parent_name:
        return ("unknown", "unknown")
    src, tgt = parent_name.split("_", 1)
    return (src, tgt)


def resolve_audio_path(audio_root: Path, raw_source: str) -> Path:
    raw_path = Path(raw_source)
    if not raw_path.is_absolute():
        raw_path = (audio_root / raw_source).resolve()
    return raw_path


def build_manifest_entries(
    src_file: Path,
    audio_root: Path,
    skip_missing_audio: bool,
) -> Tuple[RecordingSet, SupervisionSet, int, int]:
    src_lang, tgt_lang = parse_lang_pair(src_file)
    recordings: Dict[str, Recording] = {}
    supervisions = []
    total = 0
    kept = 0
    with src_file.open("r", encoding="utf-8") as inp:
        for raw_line in inp:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total += 1
            item = json.loads(raw_line)
            audio_path = resolve_audio_path(audio_root, item["source"])
            if not audio_path.is_file():
                msg = f"Missing audio: {audio_path}"
                if skip_missing_audio:
                    logging.warning("%s (skipping)", msg)
                    continue
                raise FileNotFoundError(msg)

            recording_id = audio_path.stem
            if recording_id not in recordings:
                recordings[recording_id] = Recording.from_file(
                    path=str(audio_path), recording_id=recording_id
                )

            supervision = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=recordings[recording_id].duration,
                channel=0,
                text=item.get("text", ""),
                language=src_lang,
                speaker="unknown",
                gender="",
                custom={
                    "st_text": item.get("st_text", ""),
                    "lang": tgt_lang,
                },
            )
            supervisions.append(supervision)
            kept += 1

    recording_set = RecordingSet.from_recordings(recordings.values())
    supervision_set = SupervisionSet.from_segments(supervisions)
    return recording_set, supervision_set, total, kept


def split_cached_cuts(
    cut_set: CutSet,
    feature_cache: Dict[str, Dict],
    refresh_cache: bool,
) -> Tuple[List, List]:
    cached_cuts = []
    pending_cuts = []
    for cut in cut_set:
        rid = cut.recording.id
        if not refresh_cache and rid in feature_cache:
            feat = Features.from_dict(feature_cache[rid])
            cached_cuts.append(replace(cut, features=feat))
        else:
            pending_cuts.append(cut)
    return cached_cuts, pending_cuts


def update_feature_cache(feature_cache_path: Path, cache: Dict[str, Dict]) -> None:
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def load_feature_cache(feature_cache_path: Path) -> Dict[str, Dict]:
    if not feature_cache_path.is_file():
        return {}
    with feature_cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shard_cuts_by_duration(cuts: List, max_duration: float) -> List[List]:
    """Group cuts into shards whose total duration does not exceed max_duration."""
    if max_duration <= 0:
        return [cuts]

    shards: List[List] = []
    current: List = []
    current_dur = 0.0

    for cut in cuts:
        dur = float(cut.duration or 0.0)
        if current and current_dur + dur > max_duration:
            shards.append(current)
            current = []
            current_dur = 0.0
        current.append(cut)
        current_dur += dur

    if current:
        shards.append(current)

    return [shard for shard in shards if shard]


def add_dataloading_info(cut):
    info = {"rank": 0, "world_size": 1, "worker_id": None}
    return cut.with_custom("dataloading_info", info)


def process_file(
    src_file: Path,
    dst_file: Path,
    audio_root: Path,
    storage_root: Path,
    feature_cache: Dict[str, Dict],
    feature_cache_path: Path,
    num_workers: int,
    batch_duration: float,
    device: str,
    train_shard_duration: float,
    skip_missing_audio: bool,
    refresh_cache: bool,
) -> Tuple[int, int]:
    storage_root.mkdir(parents=True, exist_ok=True)
    recordings, supervisions, total, kept = build_manifest_entries(
        src_file=src_file,
        audio_root=audio_root,
        skip_missing_audio=skip_missing_audio,
    )
    if kept == 0:
        logging.warning("No usable entries in %s; skipping.", src_file)
        return total, kept

    cut_set = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions,
    )

    cached_cuts, pending_cuts = split_cached_cuts(
        cut_set=cut_set,
        feature_cache=feature_cache,
        refresh_cache=refresh_cache,
    )

    rel = src_file.parent.relative_to(src_file.parents[1])
    base = src_file.stem
    storage_path = storage_root / rel / base
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    actual_device = device
    if device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=actual_device))

    new_cuts = pending_cuts
    should_shard = train_shard_duration > 0 and "train" in src_file.stem and new_cuts

    if new_cuts:
        if should_shard:
            shards = shard_cuts_by_duration(new_cuts, train_shard_duration)
        else:
            shards = [new_cuts]

        shard_cut_sets = []
        for idx, shard in enumerate(shards):
            shard_storage = (
                f"{storage_path}_shard{idx:04d}"
                if len(shards) > 1
                else str(storage_path)
            )
            shard_set = CutSet.from_cuts(shard)
            shard_set = shard_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=shard_storage,
                storage_type=LilcomChunkyWriter,
                num_workers=num_workers,
                batch_duration=batch_duration,
                overwrite=True,
            )
            for cut in shard_set:
                feature_cache[cut.recording.id] = cut.features.to_dict()
            shard_cut_sets.append(shard_set)

        new_cut_set = CutSet.from_cuts(chain.from_iterable(shard_cut_sets))
        cached_cuts.extend(new_cut_set)
        update_feature_cache(feature_cache_path, feature_cache)

    final_cut_set = CutSet.from_cuts(cached_cuts)

    final_cut_set = final_cut_set.map(add_dataloading_info)
    final_cut_set.to_file(dst_file)
    return total, kept


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    args.src_dir = args.src_dir.resolve()
    args.dst_dir = args.dst_dir.resolve()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    audio_root = args.audio_root.resolve()
    storage_root = args.storage_root.resolve()
    feature_cache_path = args.feature_cache.resolve()
    feature_cache = {} if args.refresh_cache else load_feature_cache(feature_cache_path)

    logging.info(
        "Converting normalized texts from %s into feature-rich CutSet manifests under %s",
        args.src_dir,
        args.dst_dir,
    )

    processed_files = 0
    total_lines = 0
    total_kept = 0
    for src_file in iter_jsonl_files(args.src_dir):
        dst_file = ensure_destination_path(src_file, args.src_dir, args.dst_dir)
        if dst_file.is_file() and not args.overwrite:
            logging.info("Skipping %s (exists)", dst_file)
            continue
        logging.info("Processing %s -> %s", src_file, dst_file)
        file_total, file_kept = process_file(
            src_file=src_file,
            dst_file=dst_file,
            audio_root=audio_root,
            storage_root=storage_root,
            feature_cache=feature_cache,
            feature_cache_path=feature_cache_path,
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            device=args.device,
            train_shard_duration=args.train_shard_duration,
            skip_missing_audio=args.skip_missing_audio,
            refresh_cache=args.refresh_cache,
        )
        processed_files += 1
        total_lines += file_total
        total_kept += file_kept
        logging.info(
            "Finished %s (lines=%d, kept=%d)",
            src_file.name,
            file_total,
            file_kept,
        )

    logging.info(
        "Done. files=%d, lines=%d, kept=%d",
        processed_files,
        total_lines,
        total_kept,
    )


if __name__ == "__main__":
    main()


"""
Example usage:

python texts_to_cuts.py \
  --src-dir ./normalizer \
  --dst-dir ./manifests \
  --audio-root /path/to/audio/root \
  --storage-root ./fbank \
  --feature-cache ./fbank/feature_cache.json \
  --num-workers 12 \
  --batch-duration 400 \
  --skip-missing-audio \
  --overwrite \
  --verbose

Options:
  --src-dir: Root directory containing normalized *.jsonl files (recursively traversed).
  --dst-dir: Output root for *_cuts.jsonl.gz manifests (mirrors the src layout).
  --audio-root: Base path that the JSON 'source' field is relative to.
  --storage-root: Root directory for storing FBANK .lca feature chunks.
  --feature-cache: JSON file for memoizing extracted features; reuses cached features
                   for previously seen audio files to avoid redundant computation.
  --refresh-cache: If specified, ignores the cache and recomputes all features.
  --num-workers / --batch-duration / --device: Control parallelism and device for
                   feature extraction (auto prefers GPU if available).
  --skip-missing-audio: Skip entries with missing audio files instead of raising an error.
  --overwrite: Overwrite existing *_cuts.jsonl.gz outputs.
  --verbose: Enable more detailed logging.
"""
