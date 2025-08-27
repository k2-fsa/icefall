#!/usr/bin/env python3
"""
Prepare CHiME-4 dataset for icefall ASR experiments.
Creates lhotse manifests for CHiME-4 audio and supervision data.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.recipes.utils import read_manifests_if_cached


def get_chime4_audio_paths(audio_root: Path) -> Dict[str, List[Path]]:
    """Get all CHiME-4 audio file paths organized by subset."""
    audio_paths = {}
    
    # Define CHiME-4 subsets
    subsets = [
        'dt05_bth', 'dt05_bus_real', 'dt05_bus_simu', 'dt05_caf_real', 'dt05_caf_simu',
        'dt05_ped_real', 'dt05_ped_simu', 'dt05_str_real', 'dt05_str_simu',
        'et05_bth', 'et05_bus_real', 'et05_bus_simu', 'et05_caf_real', 'et05_caf_simu',
        'et05_ped_real', 'et05_ped_simu', 'et05_str_real', 'et05_str_simu',
        'tr05_bth', 'tr05_bus_real', 'tr05_bus_simu', 'tr05_caf_real', 'tr05_caf_simu',
        'tr05_org', 'tr05_ped_real', 'tr05_ped_simu', 'tr05_str_real', 'tr05_str_simu'
    ]
    
    for subset in subsets:
        subset_dir = audio_root / subset
        if subset_dir.exists():
            wav_files = list(subset_dir.glob("*.wav"))
            if wav_files:
                audio_paths[subset] = wav_files
                logging.info(f"Found {len(wav_files)} files in {subset}")
    
    return audio_paths


def parse_chime4_transcription_file(trn_file: Path) -> List[Tuple[str, str]]:
    """Parse CHiME-4 transcription file and return list of (utterance_id, text) pairs."""
    transcriptions = []
    
    with open(trn_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # CHiME-4 transcription format: "text (utterance_id)"
            if line.endswith(')') and '(' in line:
                parts = line.rsplit('(', 1)
                if len(parts) == 2:
                    text = parts[0].strip()
                    utterance_id = parts[1].rstrip(')').strip()
                    transcriptions.append((utterance_id, text))
    
    return transcriptions


def get_chime4_transcriptions(transcript_root: Path) -> Dict[str, str]:
    """Get all CHiME-4 transcriptions organized by utterance ID."""
    all_transcriptions = {}
    
    # Process individual subset transcription files
    for trn_file in transcript_root.glob("*/*.trn"):
        subset_name = trn_file.parent.name
        logging.info(f"Processing transcriptions from {trn_file}")
        
        transcriptions = parse_chime4_transcription_file(trn_file)
        for utterance_id, text in transcriptions:
            all_transcriptions[utterance_id] = text
        
        logging.info(f"Added {len(transcriptions)} transcriptions from {subset_name}")
    
    # Also process .trn_all files
    for trn_all_file in transcript_root.glob("*.trn_all"):
        logging.info(f"Processing transcriptions from {trn_all_file}")
        
        transcriptions = parse_chime4_transcription_file(trn_all_file)
        for utterance_id, text in transcriptions:
            all_transcriptions[utterance_id] = text
        
        logging.info(f"Added {len(transcriptions)} transcriptions from {trn_all_file.name}")
    
    return all_transcriptions


def create_chime4_recordings(audio_paths: Dict[str, List[Path]]) -> RecordingSet:
    """Create RecordingSet from CHiME-4 audio files."""
    recordings = []
    
    for subset, wav_files in audio_paths.items():
        for wav_file in wav_files:
            # Extract utterance ID from filename
            # Example: F01_22GC010A_BTH.CH0.wav -> F01_22GC010A_BTH
            utterance_id = wav_file.stem
            if '.CH' in utterance_id:
                utterance_id = utterance_id.split('.CH')[0]
            
            try:
                recording = Recording.from_file(wav_file)
                # Create new recording with custom ID instead of using with_id()
                recording = Recording(
                    id=utterance_id,
                    sources=recording.sources,
                    sampling_rate=recording.sampling_rate,
                    num_samples=recording.num_samples,
                    duration=recording.duration,
                    channel_ids=recording.channel_ids,
                    transforms=recording.transforms
                )
                recordings.append(recording)
            except Exception as e:
                logging.warning(f"Failed to process {wav_file}: {e}")
                continue
    
    logging.info(f"Created {len(recordings)} recordings")
    return RecordingSet.from_recordings(recordings)


def create_chime4_supervisions(transcriptions: Dict[str, str], recordings: RecordingSet) -> SupervisionSet:
    """Create SupervisionSet from CHiME-4 transcriptions."""
    supervisions = []
    
    for recording in recordings:
        utterance_id = recording.id
        if utterance_id in transcriptions:
            text = transcriptions[utterance_id]
            supervision = SupervisionSegment(
                id=utterance_id,
                recording_id=utterance_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=text,
                language="English"
            )
            supervisions.append(supervision)
        else:
            logging.warning(f"No transcription found for {utterance_id}")
    
    logging.info(f"Created {len(supervisions)} supervisions")
    return SupervisionSet.from_segments(supervisions)


def prepare_chime4(
    audio_root: Path,
    transcript_root: Path,
    output_dir: Path
) -> None:
    """Prepare CHiME-4 dataset and save manifests."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get audio file paths
    logging.info("Scanning for CHiME-4 audio files...")
    audio_paths = get_chime4_audio_paths(audio_root)
    
    # Get transcriptions
    logging.info("Loading CHiME-4 transcriptions...")
    transcriptions = get_chime4_transcriptions(transcript_root)
    logging.info(f"Loaded {len(transcriptions)} transcriptions")
    
    # Create recordings
    logging.info("Creating recordings manifest...")
    recordings = create_chime4_recordings(audio_paths)
    
    # Create supervisions
    logging.info("Creating supervisions manifest...")
    supervisions = create_chime4_supervisions(transcriptions, recordings)
    
    # Create cuts
    logging.info("Creating cuts manifest...")
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    
    # Separate by evaluation sets (dt05, et05) and training (tr05)
    dt05_cuts = cuts.filter(lambda cut: cut.id.startswith('dt05') or 'dt05' in cut.id)
    et05_cuts = cuts.filter(lambda cut: cut.id.startswith('et05') or 'et05' in cut.id)
    tr05_cuts = cuts.filter(lambda cut: cut.id.startswith('tr05') or 'tr05' in cut.id)
    
    # Save manifests
    logging.info("Saving manifests...")
    
    if len(dt05_cuts) > 0:
        dt05_recordings = recordings.filter(lambda r: r.id in [c.recording.id for c in dt05_cuts])
        dt05_supervisions = supervisions.filter(lambda s: s.recording_id in [c.recording.id for c in dt05_cuts])
        
        dt05_recordings.to_file(output_dir / "chime4_recordings_dt05.jsonl.gz")
        dt05_supervisions.to_file(output_dir / "chime4_supervisions_dt05.jsonl.gz")
        dt05_cuts.to_file(output_dir / "chime4_cuts_dt05.jsonl.gz")
        logging.info(f"Saved dt05 manifests with {len(dt05_cuts)} cuts")
    
    if len(et05_cuts) > 0:
        et05_recordings = recordings.filter(lambda r: r.id in [c.recording.id for c in et05_cuts])
        et05_supervisions = supervisions.filter(lambda s: s.recording_id in [c.recording.id for c in et05_cuts])
        
        et05_recordings.to_file(output_dir / "chime4_recordings_et05.jsonl.gz")
        et05_supervisions.to_file(output_dir / "chime4_supervisions_et05.jsonl.gz")
        et05_cuts.to_file(output_dir / "chime4_cuts_et05.jsonl.gz")
        logging.info(f"Saved et05 manifests with {len(et05_cuts)} cuts")
    
    if len(tr05_cuts) > 0:
        tr05_recordings = recordings.filter(lambda r: r.id in [c.recording.id for c in tr05_cuts])
        tr05_supervisions = supervisions.filter(lambda s: s.recording_id in [c.recording.id for c in tr05_cuts])
        
        tr05_recordings.to_file(output_dir / "chime4_recordings_tr05.jsonl.gz")
        tr05_supervisions.to_file(output_dir / "chime4_supervisions_tr05.jsonl.gz")
        tr05_cuts.to_file(output_dir / "chime4_cuts_tr05.jsonl.gz")
        logging.info(f"Saved tr05 manifests with {len(tr05_cuts)} cuts")
    
    logging.info(f"CHiME-4 data preparation completed. Manifests saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CHiME-4 dataset for icefall")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/home/nas/DB/CHiME4/data/audio/16kHz/isolated"),
        help="Path to CHiME-4 audio root directory"
    )
    parser.add_argument(
        "--transcript-root", 
        type=Path,
        default=Path("/home/nas/DB/CHiME4/data/transcriptions"),
        help="Path to CHiME-4 transcription root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chime4"),
        help="Output directory for manifest files"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    
    prepare_chime4(args.audio_root, args.transcript_root, args.output_dir)


if __name__ == "__main__":
    main()
