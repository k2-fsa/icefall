import argparse
from glob import glob
from itertools import islice
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions, CutSet
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

FULL_DATA_PARTS = (
    "eval1",
    "eval2",
    "eval3",
    "core",
    "noncore",
)

DATASETS = (
    "train",
    "valid",
    "eval1",
    "eval2",
    "eval3"
)

def parse_transcript_header(line : str):
    sgid, start, end, line = line.split(' ', maxsplit=3)
    return (sgid, float(start), float(end), line)

def parse_one_recording(
    template : str,
    wavlist_path : Path, 
    recording_id : str
) -> Tuple[Recording, List[SupervisionSegment]]:
    transcripts = []
    
    for trans in glob(template + '*.txt'):
        trans_type = trans.replace(template + '-', '').replace(".txt", '')
        transcripts.append([(trans_type, t) for t in Path(trans).read_text().split('\n')])
    
    assert all(len(c) == len(transcripts[0]) for c in transcripts), transcripts
    wav = wavlist_path.read_text()
        
    recording = Recording.from_file(wav, recording_id=recording_id)
    
    supervision_segments = []
    
    for texts in zip(*transcripts):
        customs = {}
        for trans_type, text in texts:
            sgid, start, end, customs[trans_type] = parse_transcript_header(text)
        supervision_segments.append(
            SupervisionSegment(
                id=sgid,
                recording_id=recording_id,
                start=start,
                duration=(end-start),
                channel=0,
                language="Japanese",
                speaker=recording_id,
                text="",
                custom=customs
            )
        )
    
    return recording, supervision_segments

def prepare_csj(
    output_dir : Pathlike,
    trans_dir : Pathlike, 
    dataset_parts : Union[str, Sequence[str]] = FULL_DATA_PARTS, 
    num_jobs : int = 1,
    split : int = 4000
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    
    # corpus_dir = Path(corpus_dir)
    # assert corpus_dir.is_dir(), f"No such directory for corpus_dir: {corpus_dir}"
    trans_dir = Path(trans_dir)
    assert trans_dir.is_dir(), f"No such directory for trans_dir: {trans_dir}"

    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
        
    # manifests = {}
    
    if output_dir is None:
        return 
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing CSJ subset: {part}")
            # if manifests_exist(part=part, output_dir=output_dir):
            #     logging.info(f"CSJ subset: {part} already prepared - skipping.")
            
            recordings = []
            supervisions = []
            part_path = trans_dir / part
            futures = []
            
            for trans in part_path.glob("*/*-disfluent.txt"):
                template = trans.as_posix().rstrip('-disfluent.txt')
                spk = trans.name.rstrip('-disfluent.txt')
                wavlist = Path(template + '-wav.list')
                futures.append(
                    ex.submit(parse_one_recording, template, wavlist, spk)
                )
                # futures.append(parse_one_recording(template, wavlist, spk))
                # parse_one_recording(morph, pron, clean, wavlist, spk)

            # for future in futures:
            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                # result = future
                assert result
                recording, segments = result
                recordings.append(recording)
                supervisions.extend(segments)
            
            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)
            validate_recordings_and_supervisions(recording_set, supervision_set)
            
            supervision_set.to_file(output_dir / f"supervisions_{part}.json")
            recording_set.to_file(output_dir / f"recordings_{part}.json")

    logging.info(f"Creating valid cuts, split at {split}")
    # Create train and valid cuts
    recording_set = RecordingSet.from_file(output_dir / "recordings_core.json") \
        + RecordingSet.from_file(output_dir / "recordings_noncore.json")
    supervision_set = SupervisionSet.from_file(output_dir / "supervisions_core.json") \
        + SupervisionSet.from_file(output_dir / "supervisions_noncore.json")

    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    
    valid_set = CutSet.from_cuts(islice(cut_set, 0, split))
    valid_set.to_jsonl(output_dir / "cuts_valid.jsonl.gz")
    valid_set.to_json(output_dir / "cuts_valid.json")
    
    logging.info(f"Creating train cuts")
    train_set = CutSet.from_cuts(islice(cut_set, split, None))
    
    train_set = (
        train_set
        + train_set.perturb_speed(0.9)
        + train_set.perturb_speed(1.1)
    )
    train_set.to_jsonl(output_dir / "cuts_train.jsonl.gz")
    train_set.to_json(output_dir / "cuts_train.json")
    
    logging.info("Creating eval cuts.")
    # Create eval datasets
    for i in range(1, 4):
        cut_set = CutSet.from_manifests(
            recordings=RecordingSet.from_file(output_dir / f"recordings_eval{i}.json"),
            supervisions=SupervisionSet.from_file(output_dir / f"supervisions_eval{i}.json")
        )
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_set.to_jsonl(output_dir / f"cuts_eval{i}.jsonl.gz")
        cut_set.to_json(output_dir / f"cuts_eval{i}.json")

def get_args():
    #TODO: fill in parser
    parser = argparse.ArgumentParser(description="""
             TODO"""
    )
    
    parser.add_argument("--trans-dir", type=Path,
                        help="Path to transcripts")
    parser.add_argument("--manifest-dir", type=Path,
                        help="Path to save manifests")    
    parser.add_argument("--split", type=int,
                        help=(
                        "Index at which to split the train dataset. "
                        "Cuts before this index will fall under valid dataset. "
                        "Cuts after this index will fall under train dataset"
                        ))    
    parser.add_argument("--debug", action="store_true",
                        help="Use hardcoded parameters")
    
    return parser.parse_args()
    
def main():
    args = get_args()
    
    if args.debug:
        args.trans_dir = Path("/mnt/minami_data_server/t2131178/corpus/CSJ/retranscript_new")
        args.manifest_dir = Path("data/manifests")
        args.split = 4000
        
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,        
        )
    
    prepare_csj(
        # dataset_parts=['eval/eval1'],
        output_dir=args.manifest_dir,
        trans_dir=args.trans_dir,
        num_jobs=4,
        split=args.split
    )
    logging.info("Done.")
    
if __name__ == '__main__':
    main()