from pathlib import Path

from backend_np import WujiEEGBackend
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio.backend import set_current_audio_backend
from tqdm import tqdm

set_current_audio_backend(WujiEEGBackend())

SPLIT=Path("/nvme3/wyc/sleep-net-zero/index/sleep_staging/hsp_nsrr.csv")
DATA_DIR=Path("/home/jinzengrui/proj/biofall/egs/tokenizer/CODEC/data/from_wyc")

if __name__ == "__main__":
    with open(SPLIT, "r") as f:
        csv_lines = f.readlines()
    csv_lines = csv_lines[1:]
    train_cuts, val_cuts = [], []

    for line in tqdm(csv_lines):
        line = line.strip()
        npz_path, sess_id, duration, split = line.split(",")
        duration = float(duration)
        npz_path = Path(npz_path)
        npz_fname = npz_path.stem.split(".")[0]
        audio = Recording.from_file(npz_path, recording_id=f"{sess_id}-{npz_fname}")
        cut = MonoCut(
            id=f"{sess_id}-{npz_fname}",
            start=0.0,
            duration=duration,
            channel=0,
            recording=audio,
            supervisions=[
                SupervisionSegment(
                    id=f"{sess_id}-{npz_fname}",
                    recording_id=f"{sess_id}-{npz_fname}",
                    start=0.0,
                    duration=duration,
                    channel=0,
                    text="",
                    language="",
                    speaker=sess_id,
                )
            ],
        )
        if split == "train":
            train_cuts.append(cut)
        elif split == "val":
            val_cuts.append(cut)
        else:
            raise ValueError(f"Unknown split: {split}")
        
    train_cuts = CutSet.from_cuts(cuts=train_cuts)
    train_cuts.to_jsonl(DATA_DIR / "train.jsonl.gz")
    val_cuts = CutSet.from_cuts(cuts=val_cuts)
    val_cuts.to_jsonl(DATA_DIR / "val.jsonl.gz")