#!/usr/bin/env python3
"""
Simple CHiME-4 test dataloader creation script.
Creates a small subset for quick testing.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse import Fbank, FbankConfig
from torch.utils.data import DataLoader


def create_simple_chime4_test_loader(
    audio_root: Path = Path("/home/nas/DB/CHiME4/data/audio/16kHz/isolated"),
    transcript_root: Path = Path("/home/nas/DB/CHiME4/data/transcriptions"),
    max_files: int = 10
) -> DataLoader:
    """Create a simple test dataloader with limited CHiME-4 files."""
    
    logging.info(f"Creating simple CHiME-4 test loader with max {max_files} files")
    
    # Focus on dt05_bth (clean booth) for simplicity
    audio_dir = audio_root / "dt05_bth"
    transcript_dir = transcript_root / "dt05_bth"
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
    
    # Get limited audio files
    wav_files = sorted(list(audio_dir.glob("*.wav")))[:max_files]
    logging.info(f"Found {len(wav_files)} audio files to process")
    
    # Parse transcriptions from individual .trn files
    transcriptions = {}
    for trn_file in transcript_dir.glob("*.trn"):
        try:
            with open(trn_file, 'r', encoding='utf-8') as f:
                line = f.read().strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        utterance_id = parts[0]
                        text = parts[1]
                        transcriptions[utterance_id] = text
                        logging.debug(f"Loaded transcription: {utterance_id}")
        except Exception as e:
            logging.warning(f"Failed to read {trn_file}: {e}")
    
    logging.info(f"Loaded {len(transcriptions)} transcriptions")
    
    # Create recordings and supervisions
    recordings = []
    supervisions = []
    
    for wav_file in wav_files:
        # Extract utterance ID from filename (remove .CH0, etc.)
        utterance_id = wav_file.stem
        if '.CH' in utterance_id:
            utterance_id = utterance_id.split('.CH')[0]
        
        # Skip if no transcription
        if utterance_id not in transcriptions:
            logging.warning(f"No transcription for {utterance_id}")
            continue
        
        try:
            # Create recording
            recording = Recording.from_file(wav_file)
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
            
            # Create supervision
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
            
            logging.info(f"Processed: {utterance_id} - '{text[:50]}...'")
            
        except Exception as e:
            logging.warning(f"Failed to process {wav_file}: {e}")
            continue
    
    if not recordings:
        raise ValueError("No valid recordings found!")
    
    # Create manifests
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    
    logging.info(f"Created {len(cuts)} cuts for CHiME-4 test")
    
    # Create dataset and dataloader
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        return_cuts=True
    )
    
    # Simple sampler - no bucketing for test
    from lhotse.dataset import SimpleCutSampler
    sampler = SimpleCutSampler(cuts, max_duration=30.0, shuffle=False)
    
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=1
    )
    
    logging.info(f"Created CHiME-4 test dataloader with {len(cuts)} utterances")
    return dataloader, cuts


def main():
    parser = argparse.ArgumentParser(description="Create simple CHiME-4 test dataloader")
    parser.add_argument("--max-files", type=int, default=10, help="Max files to process")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    
    try:
        dataloader, cuts = create_simple_chime4_test_loader(max_files=args.max_files)
        
        # Test the dataloader
        logging.info("Testing dataloader...")
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Just test first 2 batches
                break
            logging.info(f"Batch {i}: {batch['supervisions']['text']}")
            
        logging.info("CHiME-4 test dataloader creation successful!")
        
    except Exception as e:
        logging.error(f"Failed to create CHiME-4 test dataloader: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
