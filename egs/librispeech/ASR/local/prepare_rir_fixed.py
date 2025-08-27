#!/usr/bin/env python3
"""
Fixed version of prepare RIR data for lhotse.
This script converts rir.scp file to lhotse manifest format.
"""

import argparse
import logging
from pathlib import Path
from typing import List
import json
import gzip

from lhotse import CutSet, Recording, RecordingSet
from lhotse.audio import AudioSource
from lhotse.utils import Pathlike

def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare RIR data for lhotse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--rir-scp",
        type=Path,
        required=True,
        help="Path to rir.scp file containing RIR file paths",
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        required=True,
        help="Output directory for RIR manifests",
    )
    
    return parser.parse_args()


def prepare_rir_manifest(
    rir_scp: Pathlike,
    output_dir: Pathlike,
) -> None:
    """
    Prepare RIR manifest from rir.scp file.
    
    Args:
        rir_scp: Path to rir.scp file
        output_dir: Output directory for manifests
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recordings = []
    
    # Read rir.scp file
    with open(rir_scp, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse line: either "path" or "id path"
            parts = line.split()
            if len(parts) == 1:
                rir_path = parts[0]
                rir_id = f"rir_{line_idx:06d}"
            elif len(parts) == 2:
                rir_id, rir_path = parts
            else:
                logging.warning(f"Invalid line in rir.scp: {line}")
                continue
            
            # Check if file exists
            rir_path = Path(rir_path)
            if not rir_path.exists():
                logging.warning(f"RIR file not found: {rir_path}")
                continue
            
            # Create recording
            recording = Recording(
                id=rir_id,
                sources=[
                    AudioSource(
                        type="file",
                        channels=[0],
                        source=str(rir_path.resolve()),
                    )
                ],
                sampling_rate=16000,  # Assume 16kHz, will be auto-detected by lhotse
                num_samples=None,     # Will be auto-detected
                duration=None,        # Will be auto-detected
            )
            
            recordings.append(recording)
    
    logging.info(f"Found {len(recordings)} RIR files")
    
    # Create recording set and save
    recording_set = RecordingSet.from_recordings(recordings)
    
    # Validate recordings (this will auto-detect duration, sampling_rate, etc.)
    logging.info("Validating RIR recordings...")
    
    # Save recording manifest
    output_path = output_dir / "rir_recordings.jsonl.gz"
    recording_set.to_file(output_path)
    logging.info(f"Saved RIR recording manifest to {output_path}")
    
    # Create cuts manually to ensure correct format
    logging.info("Creating RIR cuts manifest...")
    cuts_data = []
    
    for recording in recording_set:
        cut_data = {
            "id": f"{recording.id}-0",
            "start": 0,
            "duration": recording.duration,
            "channel": 0,
            "recording": recording.to_dict()
        }
        cuts_data.append(cut_data)
    
    # Save cuts manually
    cuts_output_path = output_dir / "rir_cuts.jsonl.gz"
    with gzip.open(cuts_output_path, 'wt') as f:
        for cut in cuts_data:
            f.write(json.dumps(cut) + '\n')
    
    logging.info(f"Saved RIR cuts manifest to {cuts_output_path}")
    
    # Verify the cuts can be loaded
    try:
        from lhotse import load_manifest
        cuts_test = load_manifest(cuts_output_path)
        logging.info(f"Successfully verified: loaded {len(cuts_test)} cuts")
    except Exception as e:
        logging.error(f"Failed to verify cuts: {e}")
    
    return recording_set


def main():
    args = get_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    logging.info("Preparing RIR data...")
    prepare_rir_manifest(
        rir_scp=args.rir_scp,
        output_dir=args.output_dir,
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()
