#!/usr/bin/env python3
"""
This file computes fbank features of the RIR dataset.
It looks for RIR recordings and generates fbank features.

The generated fbank features are saved in data/fbank.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
import soundfile as sf
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    MonoCut,
    RecordingSet,
    Recording,
)
from lhotse.audio import AudioSource

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_rir(
    rir_scp: str = "data/manifests/rir.scp",
    num_mel_bins: int = 80, 
    output_dir: str = "data/fbank",
    max_files: int = None
):
    """
    Compute fbank features for RIR files.
    
    Args:
        rir_scp: Path to rir.scp file
        num_mel_bins: Number of mel filter banks
        output_dir: Output directory for features  
        max_files: Maximum number of RIR files to process (for testing)
    """
    output_dir = Path(output_dir)
    num_jobs = min(15, os.cpu_count())

    rir_cuts_path = output_dir / "rir_cuts.jsonl.gz"

    if rir_cuts_path.is_file():
        logging.info(f"{rir_cuts_path} already exists - skipping")
        return

    logging.info("Extracting features for RIR")

    # Create RIR recordings from scp file
    recordings = []
    with open(rir_scp, 'r') as f:
        for idx, line in enumerate(f):
            if max_files and idx >= max_files:
                break
                
            rir_path = Path(line.strip())
            if not rir_path.exists():
                logging.warning(f"RIR file not found: {rir_path}")
                continue
                
            rir_id = f"rir_{idx:06d}"
            
            try:
                # Get audio info using soundfile
                with sf.SoundFile(rir_path) as audio_file:
                    sampling_rate = audio_file.samplerate
                    num_samples = len(audio_file)
                    duration = num_samples / sampling_rate
                
                # Create recording with proper metadata
                recording = Recording(
                    id=rir_id,
                    sources=[
                        AudioSource(
                            type="file",
                            channels=[0],
                            source=str(rir_path.resolve()),
                        )
                    ],
                    sampling_rate=int(sampling_rate),
                    num_samples=int(num_samples),
                    duration=float(duration),
                )
                recordings.append(recording)
                
            except Exception as e:
                logging.warning(f"Failed to process {rir_path}: {e}")
                continue
            
            if (idx + 1) % 1000 == 0:
                logging.info(f"Processed {idx + 1} RIR files...")

    logging.info(f"Found {len(recordings)} RIR files")

    # Create recording set
    rir_recordings = RecordingSet.from_recordings(recordings)

    # Feature extractor
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:
        # Create cuts and compute features
        rir_cuts = (
            CutSet.from_manifests(recordings=rir_recordings)
            .compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/rir_feats",
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
        )
        rir_cuts.to_file(rir_cuts_path)
        
    logging.info(f"Saved RIR cuts with features to {rir_cuts_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rir-scp",
        type=str,
        default="data/manifests/rir.scp",
        help="Path to rir.scp file. Default: data/manifests/rir.scp",
    )
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="The number of mel bins for Fbank. Default: 80",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fbank",
        help="Output directory. Default: data/fbank",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of RIR files to process (for testing). Default: None (process all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    compute_fbank_rir(
        rir_scp=args.rir_scp,
        num_mel_bins=args.num_mel_bins,
        output_dir=args.output_dir,
        max_files=args.max_files,
    )
