"""
optional TAL_CSASR(587 hours) if available(https://ai.100tal.com/dataset).
It is a mandarin-english code-switch corpus.
"""
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def text_normalize(line: str):
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/tal_mix_data_prep.sh#L52
    sed 's/Ａ/A/g' | sed 's/Ｃ/C/g' | sed 's/Ｄ/D/g' | sed 's/Ｇ/G/g' | \
    sed 's/Ｈ/H/g' | sed 's/Ｕ/U/g' | sed 's/Ｙ/Y/g' | sed 's/ａ/a/g' | \
    sed 's/Ｉ/I/g' | sed 's/#//g' | sed 's/=//g' | sed 's/；//g' | \
    sed 's/，//g' | sed 's/？//g' | sed 's/。//g' | sed 's/\///g' | \
    sed 's/！//g' | sed 's/!//g' | sed 's/\.//g' | sed 's/\?//g' | \
    sed 's/：//g' | sed 's/,//g' | sed 's/\"//g' | sed 's/://g' | \
    sed 's/@//g' | sed 's/-/ /g' | sed 's/、/ /g' | sed 's/~/ /g' | \
    sed "s/‘/\'/g" | sed 's/Ｅ/E/g' | sed "s/’/\'/g" | sed 's/《//g' | sed 's/》//g' | \
    sed "s/[ ][ ]*$//g" | sed "s/\[//g" | sed 's/、//g'
    210_40223_210_6228_1_1533298404_4812267_555 上面是一般现在对然后然后下面呢 HE IS ALWAYS FINISHIＮG
    """
    line = line.replace("Ａ", "A")
    line = line.replace("Ｃ", "C")
    line = line.replace("Ｄ", "D")
    line = line.replace("Ｇ", "G")
    line = line.replace("Ｈ", "H")
    line = line.replace("Ｕ", "U")
    line = line.replace("Ｙ", "Y")
    line = line.replace("ａ", "a")
    line = line.replace("Ｉ", "I")
    #line = re.sub(f'#|=|；|，|？|。|/|！|!|.|?|：|,|"|:|@|-|、|~|《|》|[|]|、', "", line)
    line = re.sub(f'#|=|；|，|？|。|/|！|!|\.|\?|：|,|"|:|@|-|、|~|《|》|[|]|、', "", line)
    line = line.replace("Ｅ", "E")
    line = line.replace("Ｎ", "N")
    line = line.upper()
    return line


def prepare_tal_csasr(
    corpus_dir: Pathlike,
    transcript_path,
    datatype,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    #transcript_path = corpus_dir / "TAL_CSASR" / "label"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            content = " ".join(idx_transcript[1:])
            content = text_normalize(content)
            transcript_dict[idx_transcript[0]] = content
    dataset_parts=[datatype]
    manifests = defaultdict(dict)
    for part in tqdm(
        dataset_parts,
        desc="Process tal_csasr audio, it takes about 4 minutes using 40 cpu jobs.",
    ):
        logging.info(f"Processing tal_csasr subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)

        supervisions = []
        wav_path = corpus_dir
        recordings = RecordingSet.from_dir(
            path=wav_path, pattern="*.wav", num_jobs=num_jobs
        )
        i=0
        for audio_path in wav_path.rglob("**/*.wav"):
            i+=1
            idx = audio_path.stem
            speaker = idx
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue

            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recordings.duration(idx),
                channel=0,
                language="Chinese_En",
                speaker=speaker,
                text=text.strip(),
            )
            supervisions.append(segment)
            if i%1000==0:
              print('procesing {}/{}'.format(i,len(transcript_dict)))
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"tal_csasr_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"tal_csasr_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
import sys
corpus_dir,TextLst,datatype,Outdir,num_jobs=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
prepare_tal_csasr(corpus_dir,TextLst,datatype,Outdir,int(num_jobs),)
