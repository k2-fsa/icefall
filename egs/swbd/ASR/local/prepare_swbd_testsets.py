import argparse
import logging
import os
from pathlib import Path
import tarfile
from itertools import chain
from typing import Dict, Optional, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob, resumable_download, safe_extract

import sentencepiece as spm
from filter_cuts import filter_cuts
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool


def prepare_switchboard(
    audio_dir: Pathlike,
    transcripts_dir: Optional[Pathlike] = None,
    sentiment_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    omit_silence: bool = True,
    absolute_paths: bool = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the Switchboard corpus.
    We create two manifests: one with recordings, and the other one with text supervisions.
    When ``sentiment_dir`` is provided, we create another supervision manifest with sentiment annotations.
    :param audio_dir: Path to ``LDC97S62`` package.
    :param transcripts_dir: Path to the transcripts directory (typically named "swb_ms98_transcriptions").
        If not provided, the transcripts will be downloaded.
    :param sentiment_dir: Optional path to ``LDC2020T14`` package which contains sentiment annotations
        for SWBD segments.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param omit_silence: Whether supervision segments with ``[silence]`` token should be removed or kept.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    audio_paths = check_and_rglob(audio_dir, "*.sph")
    text_paths = check_and_rglob(transcripts_dir, "*trans.text")

    groups = []
    name_to_text = {p.stem.split("-")[0]: p for p in text_paths}
    for ap in audio_paths:
        name = ap.stem.replace("sw0", "sw")
        groups.append(
            {
                "audio": ap,
                "text-0": name_to_text[f"{name}A"],
                "text-1": name_to_text[f"{name}B"],
            }
        )

    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            group["audio"], relative_path_depth=None if absolute_paths else 3
        )
        for group in groups
    )
    supervisions = SupervisionSet.from_segments(
        chain.from_iterable(
            make_segments(
                transcript_path=group[f"text-{channel}"],
                recording=recording,
                channel=channel,
                omit_silence=omit_silence,
            )
            for group, recording in zip(groups, recordings)
            for channel in [0, 1]
        )
    )

    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if sentiment_dir is not None:
        parse_and_add_sentiment_labels(sentiment_dir, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / "swbd_recordings_all.jsonl.gz")
        supervisions.to_file(output_dir / "swbd_supervisions_all.jsonl.gz")
    return {"recordings": recordings, "supervisions": supervisions}


def make_segments(
    transcript_path: Path, recording: Recording, channel: int, omit_silence: bool = True
):
    lines = transcript_path.read_text().splitlines()
    return [
        SupervisionSegment(
            id=segment_id,
            recording_id=recording.id,
            start=float(start),
            duration=round(float(end) - float(start), ndigits=8),
            channel=channel,
            text=" ".join(words),
            language="English",
            speaker=f"{recording.id}A",
        )
        for segment_id, start, end, *words in map(str.split, lines)
        if words[0] != "[silence]" or not omit_silence
    ]


def download_and_untar(
    target_dir: Pathlike = ".", force_download: bool = False, url: str = SWBD_TEXT_URL
) -> Path:
    target_dir = Path(target_dir)
    transcript_dir = target_dir / "swb_ms98_transcriptions"
    if transcript_dir.is_dir():
        return transcript_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = "switchboard_word_alignments.tar.gz"
    tar_path = target_dir / tar_name
    resumable_download(url, filename=tar_path, force_download=force_download)
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=target_dir)
    return transcript_dir


def parse_and_add_sentiment_labels(
    sentiment_dir: Pathlike, supervisions: SupervisionSet
):
    """Read 'LDC2020T14' sentiment annotations and add then to the supervision segments."""
    import pandas as pd

    # Sanity checks
    sentiment_dir = Path(sentiment_dir)
    labels = sentiment_dir / "data" / "sentiment_labels.tsv"
    assert sentiment_dir.is_dir() and labels.is_file()
    # Read the TSV as a dataframe
    df = pd.read_csv(labels, delimiter="\t", names=["id", "start", "end", "sentiment"])
    # We are going to match the segments in LDC2020T14 with the ones we already
    # parsed from ISIP transcripts. We simply look which of the existing segments
    # fall into a sentiment-annotated time span. When doing it this way, we use
    # 51773 out of 52293 available sentiment annotations, which should be okay.
    for _, row in df.iterrows():
        call_id = row["id"].split("_")[0]
        matches = list(
            supervisions.find(
                recording_id=call_id,
                start_after=row["start"] - 1e-2,
                end_before=row["end"] + 1e-2,
            )
        )
        if not matches:
            continue
        labels = row["sentiment"].split("#")
        # SupervisionSegments returned from .find() are references to the ones in the
        # SupervisionSet, so we can just modify them. We use the "custom" field
        # to add the sentiment label. Since there were multiple annotators,
        # we add all available labels and leave it up to the user to disambiguate them.
        for segment in matches:
            segment.custom = {f"sentiment{i}": label for i, label in enumerate(labels)}
