#!/usr/bin/env python3

import sys
import logging
import shutil
import lhotse
import os
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, safe_extract, urlretrieve_progress


def prepare_mucs(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :param link_previous_utt: If true adds previous utterance id to supervisions.
        Useful for reconstructing chains of utterances as they were read.
        If previous utterance was skipped from LibriTTS datasets previous_utt label is None.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = ["train", "test", "dev"]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, prefix="mucs"
        )

    # Contents of the file
    #   ;ID  |SEX| SUBSET           |MINUTES| NAME
    #   14   | F | train-clean-360  | 25.03 | ...
    #   16   | F | train-clean-360  | 25.11 | ...
    #   17   | M | train-clean-360  | 25.04 | ...
    


    for part in tqdm(dataset_parts, desc="Preparing mucs parts from espnet files"):
        
        if manifests_exist(part=part, output_dir=output_dir, prefix="mucs"):
            logging.info(f"mucs subset: {part} already prepared - skipping.")
            continue
        recordings, supervisions, _ = lhotse.kaldi.load_kaldi_data_dir(os.path.join(corpus_dir, part), sampling_rate=16000)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            supervisions.to_file(output_dir / f"mucs_supervisions_{part}.jsonl.gz")
            recordings.to_file(output_dir / f"mucs_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recordings, "supervisions": supervisions}

    return

if __name__ == "__main__":
    datapath = sys.argv[1]
    nj = int(sys.argv[2])
    savepath = sys.argv[3]
    print(datapath, nj, savepath)
    prepare_mucs(datapath, savepath, nj)