import argparse
import csv

import torch
import torchaudio
import logging
import glob
from lhotse import load_manifest, CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from argparse import ArgumentParser

from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def parse_csv(csv_file="downloads/audioset/full_train_asedata_with_duration.csv"):

    mapping = {}
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            key = "/".join(row[0].split('/')[-2:])
            mapping[key] = row[1]
    return mapping
            

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="downloads/audioset"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="balanced",
        choices=["balanced", "unbalanced", "eval", "eval_all"]
    )
    
    parser.add_argument(
        "--feat-output-dir",
        type=str,
        default="data/fbank_audioset",
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    split = args.split
    feat_output_dir = args.feat_output_dir
    
    num_jobs = 15
    num_mel_bins = 80

    import pdb; pdb.set_trace()
    if split in ["balanced", "unbalanced"]:
        csv_file = "downloads/audioset/full_train_asedata_with_duration.csv"
    elif split == "eval":
        csv_file = "downloads/audioset/eval.csv"
    elif split == "eval_all":
        csv_file = "downloads/audioset/eval_all.csv"
    else:
        raise ValueError()

    labels = parse_csv(csv_file)

    audio_files = glob.glob(f"{dataset_dir}/eval/wav_all/*.wav")
    
    new_cuts = []
    for i, audio in enumerate(audio_files):
        cut_id = "/".join(audio.split('/')[-2:])
        recording = Recording.from_file(audio, cut_id)
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )
        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut.recording.id,
            start=0.0,
            channel=0,
            duration=cut.duration,
        )
        try:
            supervision.audio_event = labels[cut_id]
        except KeyError:
            logging.info(f"No labels found for {cut_id}.")
            supervision.audio_event = ""
        cut.supervisions = [supervision]
        new_cuts.append(cut)
        
        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")

    cuts = CutSet.from_cuts(new_cuts)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    logging.info(f"Computing fbank features for {split}")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{feat_output_dir}/{split}_{args.split}_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
    
    manifest_output_dir = feat_output_dir + "/" + f"cuts_audioset_{split}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()