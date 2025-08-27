#!/usr/bin/env python3
"""
Super simple RIR cuts creator - manual approach without complex lhotse logic
"""

import argparse
import logging
from pathlib import Path
import json
import gzip
import wave
import soundfile as sf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir-scp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=1000)
    return parser.parse_args()

def main():
    args = get_args()
    
    logging.basicConfig(level=logging.INFO)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    recordings = []
    cuts = []
    
    with open(args.rir_scp, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= args.max_files:
                break
                
            rir_path = Path(line.strip())
            if not rir_path.exists():
                continue
                
            try:
                # Use soundfile to get audio info
                info = sf.info(rir_path)
                duration = info.duration
                sampling_rate = info.samplerate
                num_samples = info.frames
                
                rir_id = f"rir_{idx:06d}"
                
                # Recording entry - same format as LibriSpeech
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
                recordings.append(recording)
                
                # Cut entry - same format as LibriSpeech
                cut = {
                    "id": f"{rir_id}-0",
                    "start": 0.0,
                    "duration": float(duration), 
                    "channel": 0,
                    "recording_id": rir_id
                }
                cuts.append(cut)
                
                if (idx + 1) % 100 == 0:
                    logging.info(f"Processed {idx + 1} files...")
                    
            except Exception as e:
                logging.warning(f"Failed {rir_path}: {e}")
                continue
    
    logging.info(f"Created {len(recordings)} recordings and {len(cuts)} cuts")
    
    # Save files
    rec_path = args.output_dir / "rir_recordings.jsonl.gz"
    with gzip.open(rec_path, 'wt') as f:
        for rec in recordings:
            f.write(json.dumps(rec) + '\n')
    
    cuts_path = args.output_dir / "rir_cuts.jsonl.gz"  
    with gzip.open(cuts_path, 'wt') as f:
        for cut in cuts:
            f.write(json.dumps(cut) + '\n')
    
    logging.info(f"Saved to {rec_path} and {cuts_path}")
    
    # Test loading
    try:
        from lhotse import load_manifest
        test_cuts = load_manifest(cuts_path)
        test_recs = load_manifest(rec_path)
        logging.info(f"✓ SUCCESS: {len(test_cuts)} cuts, {len(test_recs)} recordings loaded!")
    except Exception as e:
        logging.error(f"✗ FAILED: {e}")

if __name__ == "__main__":
    main()
