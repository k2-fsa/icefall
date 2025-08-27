#!/usr/bin/env python3
"""
Simple approach: create minimal RIR cuts without extra validation.
"""

import argparse
import logging
from pathlib import Path
import json
import gzip
import soundfile as sf

def get_args():
    parser = argparse.ArgumentParser(
        description="Create minimal RIR cuts manifest",
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
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of RIR files to process (for testing)",
    )
    
    return parser.parse_args()


def create_minimal_rir_cuts(
    rir_scp: Path,
    output_dir: Path,
    max_files: int = 1000
) -> None:
    """
    Create a minimal RIR cuts manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cuts_data = []
    recordings_data = []
    
    # Read rir.scp file (limited for testing)
    with open(rir_scp, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx >= max_files:
                break
                
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            rir_path = Path(line.strip())
            if not rir_path.exists():
                logging.warning(f"RIR file not found: {rir_path}")
                continue
                
            rir_id = f"rir_{line_idx:06d}"
            
            try:
                # Get audio info
                with sf.SoundFile(rir_path) as audio:
                    sampling_rate = audio.samplerate
                    num_samples = len(audio)
                    duration = num_samples / sampling_rate
                    
                # Create recording entry
                recording = {
                    "id": rir_id,
                    "sources": [{
                        "type": "file",
                        "channels": [0],
                        "source": str(rir_path.resolve())
                    }],
                    "sampling_rate": int(sampling_rate),
                    "num_samples": int(num_samples),
                    "duration": float(duration),
                    "channel_ids": [0]
                }
                recordings_data.append(recording)
                
                # Create cut entry
                cut = {
                    "id": f"{rir_id}-0",
                    "start": 0.0,
                    "duration": float(duration),
                    "channel": 0,
                    "recording_id": rir_id
                }
                cuts_data.append(cut)
                
                if (line_idx + 1) % 100 == 0:
                    logging.info(f"Processed {line_idx + 1} RIR files...")
                    
            except Exception as e:
                logging.warning(f"Failed to process {rir_path}: {e}")
                continue
    
    logging.info(f"Successfully processed {len(cuts_data)} RIR files")
    
    # Save recordings manifest
    recordings_path = output_dir / "rir_recordings.jsonl.gz"
    with gzip.open(recordings_path, 'wt') as f:
        for recording in recordings_data:
            f.write(json.dumps(recording) + '\n')
    logging.info(f"Saved recordings to {recordings_path}")
    
    # Save cuts manifest
    cuts_path = output_dir / "rir_cuts.jsonl.gz"
    with gzip.open(cuts_path, 'wt') as f:
        for cut in cuts_data:
            f.write(json.dumps(cut) + '\n')
    logging.info(f"Saved cuts to {cuts_path}")
    
    # Test loading
    try:
        from lhotse import load_manifest
        cuts_test = load_manifest(cuts_path)
        recordings_test = load_manifest(recordings_path)
        logging.info(f"✓ Successfully verified: {len(cuts_test)} cuts, {len(recordings_test)} recordings")
    except Exception as e:
        logging.error(f"✗ Verification failed: {e}")


def main():
    args = get_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    logging.info(f"Creating minimal RIR manifest (max {args.max_files} files)...")
    create_minimal_rir_cuts(
        rir_scp=args.rir_scp,
        output_dir=args.output_dir,
        max_files=args.max_files
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()
