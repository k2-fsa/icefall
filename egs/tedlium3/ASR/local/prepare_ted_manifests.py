import sys
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike, safe_extract
from tqdm.auto import tqdm

def tqdm_urlretrieve_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> from urllib.request import urlretrieve
    >>> with tqdm(...) as t:
    ...     reporthook = tqdm_urlretrieve_hook(t)
    ...     urlretrieve(..., reporthook=reporthook)
    Source: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to

def urlretrieve_progress(url, filename=None, data=None, desc=None):
    """
    Works exactly like urllib.request.urlretrieve, but attaches a tqdm hook to
    display a progress bar of the download.
    Use "desc" argument to display a user-readable string that informs what is
    being downloaded.
    """
    from urllib.request import urlretrieve

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)

def prepare_tedlium(
    tedlium_root: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the TED-LIUM v3 corpus.

    The manifests are created in a dict with three splits: train, dev and test.
    Each split contains a RecordingSet and SupervisionSet in a dict under keys 'recordings' and 'supervisions'.

    :param tedlium_root: Path to the unpacked TED-LIUM data.
    :return: A dict with standard corpus splits containing the manifests.
    """
    tedlium_root = Path(tedlium_root)
    output_dir = Path(output_dir) if output_dir is not None else None
    corpus = {}
    for split in ("train", "dev", "test"):
        root = tedlium_root / "legacy" / split
        recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in (root / "sph").glob("*.sph")
        )
        stms = list((root / "stm").glob("*.stm"))
        assert len(stms) == len(recordings), (
            f"Mismatch: found {len(recordings)} "
            f"sphere files and {len(stms)} STM files. "
            f"You might be missing some parts of TEDLIUM..."
        )
        segments = []
        for p in stms:
            with p.open() as f:
                for idx, l in enumerate(f):
                    rec_id, _, _, start, end, _, *words = l.split()
                    start, end = float(start), float(end)
                    text = " ".join(words).replace("{NOISE}", "[NOISE]")
                    text = text.replace(" '", "'")
                    if text == "ignore_time_segment_in_scoring":
                        continue
                    segments.append(
                        SupervisionSegment(
                            id=f"{rec_id}-{idx}",
                            recording_id=rec_id,
                            start=start,
                            duration=round(end - start, ndigits=8),
                            channel=0,
                            text=text,
                            language="English",
                            speaker=rec_id,
                        )
                    )
        supervisions = SupervisionSet.from_segments(segments)
        corpus[split] = {"recordings": recordings, "supervisions": supervisions}

        validate_recordings_and_supervisions(**corpus[split])

        if output_dir is not None:
            recordings.to_file(output_dir / f"tedlium_recordings_{split}.jsonl.gz")
            supervisions.to_file(output_dir / f"tedlium_supervisions_{split}.jsonl.gz")

    return corpus

dl_dir = sys.argv[1]
output_dir = sys.argv[2]
prepare_tedlium(dl_dir, output_dir)
