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
Utility script to sanity-check Europarl-ST CutSet manifests.

It walks through all *.jsonl.gz files under the provided directory,
runs Lhotse's built-in validators, and reports which manifests fail.

Optionally it can read the underlying audio/features (`--read-data`)
to catch mismatches between metadata and stored tensors (slow but thorough),
and with `--per-cut-details` it will report how many individual cuts fail per manifest.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lhotse import CutSet
from lhotse.qa import validate

DEFAULT_MANIFEST_ROOT = Path(__file__).resolve().parent.parent / "manifests"


@dataclass
class ValidationResult:
    path: Path
    status: str
    num_cuts: Optional[int] = None
    error: Optional[str] = None
    bad_cut_count: Optional[int] = None
    bad_cut_samples: Optional[List[str]] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Europarl-ST CutSet manifests produced by texts_to_cuts.py"
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Directory that contains *.jsonl.gz CutSet manifests.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl.gz",
        help="Glob-style pattern (relative to --manifests-dir) for files to validate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of files to inspect (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="How many parallel worker processes to spawn. Reading features/audio is I/O bound, "
        "so >1 workers can shorten runtime.",
    )
    parser.add_argument(
        "--read-data",
        action="store_true",
        help=(
            "Load audio/features referenced by the manifests while validating. "
            "This is significantly slower but detects stale feature caches."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (per-file progress).",
    )
    parser.add_argument(
        "--per-cut-details",
        action="store_true",
        help=(
            "Iterate through every cut and count the number of failures per manifest. "
            "When combined with --read-data this reveals exactly how many cuts have stale features."
        ),
    )
    parser.add_argument(
        "--bad-cut-samples",
        type=int,
        default=5,
        help="When --per-cut-details is enabled, how many failing cut IDs to show per manifest.",
    )
    return parser.parse_args()


def find_manifest_files(root: Path, pattern: str) -> List[Path]:
    files = sorted(root.rglob(pattern))
    return files


def _collect_bad_cuts(
    cuts: CutSet, read_data: bool, sample_limit: int
) -> Tuple[int, List[str], Optional[str]]:
    bad_count = 0
    samples: List[str] = []
    first_error: Optional[str] = None
    for cut in cuts:
        try:
            validate(cut, read_data=read_data)
        except AssertionError as exc:
            bad_count += 1
            msg = str(exc)
            if first_error is None:
                first_error = msg
            if len(samples) < sample_limit:
                samples.append(f"{cut.id}: {msg}")
    return bad_count, samples, first_error


def validate_manifest(
    path: Path, read_data: bool, per_cut_details: bool, sample_limit: int
) -> ValidationResult:
    cuts: Optional[CutSet] = None
    try:
        cuts = CutSet.from_file(path)
        num_cuts = len(cuts)
        if per_cut_details:
            bad_count, samples, first_error = _collect_bad_cuts(
                cuts=cuts, read_data=read_data, sample_limit=sample_limit
            )
            if bad_count > 0:
                return ValidationResult(
                    path=path,
                    status="invalid",
                    num_cuts=num_cuts,
                    error=first_error or "Per-cut validation failed.",
                    bad_cut_count=bad_count,
                    bad_cut_samples=samples,
                )
        validate(cuts, read_data=read_data)
        return ValidationResult(path=path, status="ok", num_cuts=num_cuts)
    except AssertionError as exc:
        return ValidationResult(
            path=path,
            status="invalid",
            num_cuts=len(cuts) if cuts is not None else None,
            error=str(exc),
        )
    except Exception as exc:  # pylint: disable=broad-except
        return ValidationResult(
            path=path,
            status="error",
            num_cuts=len(cuts) if cuts is not None else None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _validate_worker_wrapper(
    path_str: str, read_data: bool, per_cut_details: bool, sample_limit: int
) -> ValidationResult:
    path = Path(path_str)
    return validate_manifest(
        path,
        read_data=read_data,
        per_cut_details=per_cut_details,
        sample_limit=sample_limit,
    )


def run_serial(
    files: List[Path],
    read_data: bool,
    verbose: bool,
    per_cut_details: bool,
    sample_limit: int,
) -> List[ValidationResult]:
    results: List[ValidationResult] = []
    for idx, path in enumerate(files, start=1):
        if verbose:
            logging.info("Validating [%d/%d]: %s", idx, len(files), path)
        results.append(
            validate_manifest(
                path=path,
                read_data=read_data,
                per_cut_details=per_cut_details,
                sample_limit=sample_limit,
            )
        )
    return results


def run_parallel(
    files: List[Path],
    read_data: bool,
    verbose: bool,
    num_workers: int,
    per_cut_details: bool,
    sample_limit: int,
) -> List[ValidationResult]:
    results: List[ValidationResult] = []
    order = {path: idx for idx, path in enumerate(files)}
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _validate_worker_wrapper,
                str(path),
                read_data,
                per_cut_details,
                sample_limit,
            ): path
            for path in files
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                result = ValidationResult(
                    path=path, status="error", error=f"Worker crashed: {exc}"
                )
            results.append(result)
            if verbose:
                logging.info(
                    "Validated [%d/%d]: %s (%s)", idx, len(files), path, result.status
                )
    # Preserve original ordering for readability
    results.sort(key=lambda res: order.get(res.path, len(order)))
    return results


def summarize(results: List[ValidationResult]) -> None:
    total = len(results)
    ok = sum(res.ok for res in results)
    invalid = [res for res in results if res.status == "invalid"]
    errored = [res for res in results if res.status == "error"]

    logging.info("Checked %d manifest files.", total)
    logging.info("  OK      : %d", ok)
    logging.info("  Invalid : %d", len(invalid))
    logging.info("  Errors  : %d", len(errored))

    if invalid:
        logging.warning("First %d invalid files:", min(len(invalid), 20))
        for res in invalid[:20]:
            extra = ""
            if res.bad_cut_count is not None:
                extra = f" | bad cuts: {res.bad_cut_count}"
                if res.bad_cut_samples:
                    sample_preview = "; ".join(res.bad_cut_samples)
                    extra += f" | samples: {sample_preview}"
            logging.warning("  %s -> %s%s", res.path, res.error, extra)
    if errored:
        logging.error("First %d error files:", min(len(errored), 20))
        for res in errored[:20]:
            logging.error("  %s -> %s", res.path, res.error)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    manifests_dir = args.manifests_dir.resolve()
    if not manifests_dir.is_dir():
        raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")

    files = find_manifest_files(manifests_dir, args.pattern)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        logging.warning(
            "No manifests matching pattern '%s' under %s", args.pattern, manifests_dir
        )
        return

    logging.info("Found %d manifest files to validate.", len(files))
    if args.num_workers > 1:
        results = run_parallel(
            files=files,
            read_data=args.read_data,
            verbose=args.verbose,
            num_workers=args.num_workers,
            per_cut_details=args.per_cut_details,
            sample_limit=args.bad_cut_samples,
        )
    else:
        results = run_serial(
            files=files,
            read_data=args.read_data,
            verbose=args.verbose,
            per_cut_details=args.per_cut_details,
            sample_limit=args.bad_cut_samples,
        )

    summarize(results)


if __name__ == "__main__":
    main()

"""
Example usage:

python check_manifests.py \
  --manifests-dir ./manifests \
  --num-workers 8 \
  --read-data \
  --verbose
"""
