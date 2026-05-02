from lhotse import load_manifest
from pathlib import Path
import os

# --- Configuration ---
supervisions_path = Path("data/manifests/grid_supervisions.jsonl.gz")
recordings_path = Path("data/manifests/grid_recordings.jsonl.gz")
output_dir = Path("data/manifests")
# --- Unseen speaker setup ---
test_speakers = {"s1", "s2", "s20", "s22"}

def video_exists(recording):
    return all(os.path.exists(source.source) for source in recording.sources)

recordings = load_manifest(recordings_path)
recordings = recordings.filter(video_exists)

supervisions = load_manifest(supervisions_path)

# Split based on speaker ID extracted from recording ID
def get_speaker_id(rec_id):
    return rec_id.split("_")[0]

train_recordings = recordings.filter(
    lambda rec: get_speaker_id(rec.id) not in test_speakers
)
test_recordings = recordings.filter(
    lambda rec: get_speaker_id(rec.id) in test_speakers
)

train_recordings.to_file(output_dir / "grid_recordings_train.jsonl.gz")
test_recordings.to_file(output_dir / "grid_recordings_test.jsonl.gz")

# Split supervisions based on matching recording IDs
train_rec_ids = set(r.id for r in train_recordings)
test_rec_ids = set(r.id for r in test_recordings)

train_supervisions = supervisions.filter(lambda s: s.recording_id in train_rec_ids)
test_supervisions = supervisions.filter(lambda s: s.recording_id in test_rec_ids)

train_supervisions.to_file(output_dir / "grid_supervisions_train.jsonl.gz")
test_supervisions.to_file(output_dir / "grid_supervisions_test.jsonl.gz")

print(f"Done splitting RecordingSet by speaker.")
print(f"Train recordings: {len(train_recordings)}")
print(f"Test recordings: {len(test_recordings)}")