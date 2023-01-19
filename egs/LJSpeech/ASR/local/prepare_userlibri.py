import logging
import os
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    is_module_available,
    safe_extract,
    urlretrieve_progress,
)

# LIBRISPEECH_ALIGNMENTS_URL = (
#     "https://drive.google.com/uc?id=1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE"
# )


# def download_librispeech(
#     target_dir: Pathlike = ".",
#     dataset_parts: Optional[Union[str, Sequence[str]]] = "mini_librispeech",
#     force_download: bool = False,
#     alignments: bool = False,
#     base_url: str = "http://www.openslr.org/resources",
#     alignments_url: str = LIBRISPEECH_ALIGNMENTS_URL,
# ) -> Path:
#     """
#     Download and untar the dataset, supporting both LibriSpeech and MiniLibrispeech
#     :param target_dir: Pathlike, the path of the dir to storage the dataset.
#     :param dataset_parts: "librispeech", "mini_librispeech",
#         or a list of splits (e.g. "dev-clean") to download.
#     :param force_download: Bool, if True, download the tars no matter if the tars exist.
#     :param alignments: should we download the alignments. The original source is:
#         https://github.com/CorentinJ/librispeech-alignments
#     :param base_url: str, the url of the OpenSLR resources.
#     :param alignments_url: str, the url of LibriSpeech word alignments
#     :return: the path to downloaded and extracted directory with data.
#     """
#     target_dir = Path(target_dir)
#     corpus_dir = target_dir / "LibriSpeech"
#     target_dir.mkdir(parents=True, exist_ok=True)

#     if dataset_parts == "librispeech":
#         dataset_parts = LIBRISPEECH
#     elif dataset_parts == "mini_librispeech":
#         dataset_parts = MINI_LIBRISPEECH
#     elif isinstance(dataset_parts, str):
#         dataset_parts = [dataset_parts]

#     for part in tqdm(dataset_parts, desc="Downloading LibriSpeech parts"):
#         logging.info(f"Processing split: {part}")
#         # Determine the valid URL for a given split.
#         if part in LIBRISPEECH:
#             url = f"{base_url}/12"
#         elif part in MINI_LIBRISPEECH:
#             url = f"{base_url}/31"
#         else:
#             logging.warning(f"Invalid dataset part name: {part}")
#             continue
#         # Split directory exists and seem valid? Skip this split.
#         part_dir = corpus_dir / part
#         completed_detector = part_dir / ".completed"
#         if completed_detector.is_file():
#             logging.info(f"Skipping {part} because {completed_detector} exists.")
#             continue
#         # Maybe-download the archive.
#         tar_name = f"{part}.tar.gz"
#         tar_path = target_dir / tar_name
#         if force_download or not tar_path.is_file():
#             urlretrieve_progress(
#                 f"{url}/{tar_name}", filename=tar_path, desc=f"Downloading {tar_name}"
#             )
#         # Remove partial unpacked files, if any, and unpack everything.
#         shutil.rmtree(part_dir, ignore_errors=True)
#         with tarfile.open(tar_path) as tar:
#             safe_extract(tar, path=target_dir)
#         completed_detector.touch()

#     if alignments:
#         completed_detector = target_dir / ".ali_completed"
#         if completed_detector.is_file() and not force_download:
#             return corpus_dir
#         assert is_module_available(
#             "gdown"
#         ), 'To download LibriSpeech alignments, please install "pip install gdown"'
#         import gdown

#         ali_zip_path = str(target_dir / "LibriSpeech-Alignments.zip")
#         gdown.download(alignments_url, output=ali_zip_path)
#         with zipfile.ZipFile(ali_zip_path) as f:
#             f.extractall(path=target_dir)
#             completed_detector.touch()

#     return corpus_dir


def prepare_userlibri(
    corpus_dir: str,
    dataset_parts: str = "auto",
    output_dir: str = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    
    # corpus_audio_dir = Path(corpus_dir + "/audio_data")
    # corpus_lm_dir = Path(corpus_dir + "/lm_data")
    # corpus_dir = Path(corpus_dir)
    corpus_dir = Path(corpus_dir + "/audio_data")
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    spkwise_parent = corpus_dir / "speaker-wise-test"
    spks = os.listdir(spkwise_parent)

    bookwise_parent = corpus_dir / "book-wise-test"
    books = os.listdir(bookwise_parent)

    spks_parts = (
        set(spks)
    )
    books_parts = (
        set(books)
    )

    manifests = {}

    for s_or_b, dataset_parts in zip(["speaker-wise-test", "book-wise-test"], [spks_parts, books_parts]):
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Maybe the manifests already exist: we can read them and save a bit of preparation time.
            manifests = read_manifests_if_cached(
                dataset_parts=dataset_parts, output_dir=output_dir
            )

        with ThreadPoolExecutor(num_jobs) as ex:
            for part in tqdm(dataset_parts, desc="Dataset parts"):
                logging.info(f"Processing UserLibri subset: {part}")
                if manifests_exist(part=part, output_dir=output_dir):
                    logging.info(f"UserLibri subset: {part} already prepared - skipping.")
                    continue
                recordings = []
                supervisions = []
                part_path = corpus_dir / s_or_b / part
                futures = []
                for trans_path in tqdm(
                    part_path.rglob("*.trans.txt"), desc="Distributing tasks", leave=False
                ):
                    alignments = {}
                    with open(trans_path) as f:
                        for line in f:
                            futures.append(
                                ex.submit(parse_utterance, trans_path.parent, line, alignments)
                            )

                for future in tqdm(futures, desc="Processing", leave=False):
                    result = future.result()
                    if result is None:
                        continue
                    recording, segment = result
                    recordings.append(recording)
                    supervisions.append(segment)

                recording_set = RecordingSet.from_recordings(recordings)
                supervision_set = SupervisionSet.from_segments(supervisions)

                validate_recordings_and_supervisions(recording_set, supervision_set)

                if output_dir is not None:
                    supervision_set.to_file(
                        output_dir / f"userlibri_supervisions_{part}.jsonl.gz"
                    )
                    recording_set.to_file(
                        output_dir / f"userlibri_recordings_{part}.jsonl.gz"
                    )

                manifests[part] = {
                    "recordings": recording_set,
                    "supervisions": supervision_set,
                }

    return manifests


def parse_utterance(
    dataset_split_path: Path,
    line: str,
    alignments: Dict[str, List[AlignmentItem]],
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # Create the Recording first
    audio_path = (
        dataset_split_path
        / f"{recording_id}.flac"
    )
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="English",
        speaker=re.sub(r"-.*", r"", recording.id),
        text=text.strip(),
        alignment={"word": alignments[recording_id]}
        if recording_id in alignments
        else None,
    )
    return recording, segment


def parse_alignments(ali_path: Pathlike) -> Dict[str, List[AlignmentItem]]:
    alignments = {}
    for line in Path(ali_path).read_text().splitlines():
        utt_id, words, timestamps = line.split()
        words = words.replace('"', "").split(",")
        timestamps = [0.0] + list(map(float, timestamps.replace('"', "").split(",")))
        alignments[utt_id] = [
            AlignmentItem(
                symbol=word, start=start, duration=round(end - start, ndigits=8)
            )
            for word, start, end in zip(words, timestamps, timestamps[1:])
        ]
    return alignments

def main():
    nj = 15
    output_dir = "data/manifests"
    corpus_dir = "/DB/UserLibri"

    prepare_userlibri(corpus_dir, "auto", output_dir, nj)

main()