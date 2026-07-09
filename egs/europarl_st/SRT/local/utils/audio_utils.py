"""
Audio conversion helpers for Europarl-ST preprocessing.

Currently exposes `audio_to_flac`, a thin wrapper around FFmpeg that
extracts (and optionally trims) segments from the original recordings
while resampling to the desired rate.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


class AudioConversionError(RuntimeError):
    """Raised when FFmpeg cannot convert the provided audio snippet."""


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for `path` if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def audio_to_flac(
    input_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    sample_rate: int,
    segment_start: Optional[str] = None,
    segment_end: Optional[str] = None,
) -> None:
    """
    Convert `input_path` audio into a FLAC file at `output_path`.

    Args:
        input_path: Source audio file (.m4a in the Europarl-ST dataset).
        output_path: Destination path for the trimmed/resampled FLAC.
        sample_rate: Target sample rate in Hz (e.g., 16000).
        segment_start: Optional HH:MM:SS.sss start timestamp.
        segment_end: Optional HH:MM:SS.sss end timestamp.

    Raises:
        FileNotFoundError: If the source audio does not exist.
        AudioConversionError: When FFmpeg fails.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Audio source not found: {input_path}")

    _ensure_parent_dir(output_path)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]

    if segment_start is not None:
        cmd += ["-ss", str(segment_start)]

    if segment_end is not None:
        cmd += ["-to", str(segment_end)]

    cmd += [
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_path),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        encoding="utf-8",
        errors="ignore",
    )

    if proc.returncode != 0:
        raise AudioConversionError(
            f"FFmpeg failed ({proc.returncode}) while converting {input_path} -> {output_path}:\n{proc.stderr}"
        )
