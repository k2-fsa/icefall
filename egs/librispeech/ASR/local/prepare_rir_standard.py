#!/usr/bin/env python3
"""
Create RIR cuts using lhotse's standard approach.
This should create a properly formatted cuts manifest.
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, Recording, RecordingSet
from lhotse.audio import AudioSource
from lhotse.utils import Pathlike

def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare RIR data for lhotse using standard approach",
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
    Prepare RIR manifest using lhotse's standard approach.
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
                sampling_rate=16000,  # Will be auto-detected
                num_samples=None,     # Will be auto-detected
                duration=None,        # Will be auto-detected
            )
            
            recordings.append(recording)
    
    logging.info(f"Found {len(recordings)} RIR files")
    
    # Create recording set and validate
    recording_set = RecordingSet.from_recordings(recordings)
    
    # Save recording manifest
    recordings_output_path = output_dir / "rir_recordings.jsonl.gz"
    recording_set.to_file(recordings_output_path)
    logging.info(f"Saved RIR recording manifest to {recordings_output_path}")
    
    # Create cuts using lhotse's standard method
    logging.info("Creating RIR cuts manifest using lhotse's standard method...")
    cuts = CutSet.from_manifests(recordings=recording_set)
    
    # Save cuts manifest
    cuts_output_path = output_dir / "rir_cuts.jsonl.gz"
    cuts.to_file(cuts_output_path)
    logging.info(f"Saved RIR cuts manifest to {cuts_output_path}")
    
    # Verify the cuts can be loaded
    try:
        from lhotse import load_manifest
        cuts_test = load_manifest(cuts_output_path)
        logging.info(f"✓ Successfully verified: loaded {len(cuts_test)} cuts")
        logging.info(f"First cut ID: {cuts_test[0].id}")
        logging.info(f"First cut keys: {list(cuts_test[0].to_dict().keys())}")
    except Exception as e:
        logging.error(f"✗ Failed to verify cuts: {e}")
        
        # Try CutSet.from_file as fallback
        try:
            cuts_test2 = CutSet.from_file(cuts_output_path)
            logging.info(f"✓ CutSet.from_file worked: loaded {len(cuts_test2)} cuts")
        except Exception as e2:
            logging.error(f"✗ CutSet.from_file also failed: {e2}")
    
    return recording_set, cuts


def main():
    args = get_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    logging.info("Preparing RIR data using lhotse standard approach...")
    prepare_rir_manifest(
        rir_scp=args.rir_scp,
        output_dir=args.output_dir,
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()
