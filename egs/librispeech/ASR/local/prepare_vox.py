import logging
import sys
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

def prepare_LJSpeech(
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

    assert os.path.exists(corpus_dir), f"{corpus_dir} does not exist"

    # wav_dir = Path(corpus_dir + "/wavs")
    # wavs = os.listdir(wav_dir)

    # text_dir = Path(corpus_dir + "/wavs")
    # texts = os.listdir(text_dir)

    # wavs_parts = (
    #     set(wavs)
    # )
    # books_parts = (
    #     set(texts)
    # )

    manifests = {}

    #dataset_parts = ["train", "dev", "test"]
    dataset_parts = ["4446"]
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    import glob

    futures = []
    for part in tqdm(dataset_parts, desc="Dataset parts"):
        logging.info(f"Processing LJSpeech subset: {part}")
        if manifests_exist(part=part, output_dir=output_dir):
            logging.info(f"LJSpeech subset: {part} already prepared - skipping.")
            continue
        recordings = []
        supervisions = []
        part_path = Path(os.path.join(corpus_dir, "wavs", part))
        part_file_names = list(map(lambda x: x.strip('.wav'),os.listdir(part_path))) 
        txt_path = os.path.join(corpus_dir, "texts")
        futures = []
        
        for trans_path in tqdm(
            glob.iglob(str(txt_path) + "/*.txt"), desc="Distributing tasks", leave=False
        ):
            alignments = {}
            with open(trans_path) as f:
                cur_file_name = trans_path.split('/')[-1].replace('.txt', '')
                if cur_file_name not in part_file_names:
                    continue
                for line in f:
                    futures.append(
                        parse_utterance(part_path, trans_path + ' ' + line, alignments)
                    )

        for future in tqdm(futures, desc="Processing", leave=False):
            result = future
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
                output_dir / f"LJSpeech_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"LJSpeech_recordings_{part}.jsonl.gz"
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
    recording_id = recording_id.split('/')[-1].split('.txt')[0]

    # Create the Recording first
    audio_path = (
        dataset_split_path / f"{recording_id}.wav"
    )

    if not os.path.exists(audio_path):
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

def main(corpus_dir):
    nj = 15
    output_dir = "data/manifests"

    prepare_vox(corpus_dir, "auto", output_dir, nj)

corpus_dir = sys.argv[1]
main(corpus_dir)
