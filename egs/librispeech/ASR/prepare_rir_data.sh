#!/bin/bash

# prepare_rir_data.sh
# Script to prepare RIR data for icefall training

set -euo pipefail

stage=0
stop_stage=100

# Directories and files
rir_scp="/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/manifests/rir.scp"  # Path to your rir.scp file
data_dir="data/rir"
rir_cuts_manifest="$data_dir/rir_cuts.jsonl.gz"

. shared/parse_options.sh || exit 1

if [ $# != 1 ]; then
  echo "Usage: $0 <rir-scp-path>"
  echo "e.g.: $0 /path/to/your/rir.scp"
  echo ""
  echo "Options:"
  echo "  --stage <stage>                 # Stage to start from (default: 0)"
  echo "  --stop-stage <stop-stage>       # Stage to stop at (default: 100)"
  echo "  --data-dir <dir>                # Output directory (default: data/rir)"
  exit 1
fi

rir_scp=$1

if [ ! -f "$rir_scp" ]; then
  echo "Error: RIR scp file not found: $rir_scp"
  exit 1
fi

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Preparing RIR manifest from $rir_scp"
  
  mkdir -p $data_dir
  
  python local/prepare_rir.py \
    --rir-scp $rir_scp \
    --output-dir $data_dir
  
  log "RIR manifest saved to $rir_cuts_manifest"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Validating RIR manifest"
  
  if [ ! -f "$rir_cuts_manifest" ]; then
    echo "Error: RIR cuts manifest not found: $rir_cuts_manifest"
    exit 1
  fi
  
  # Count number of RIR files
  python -c "
from lhotse import load_manifest
cuts = load_manifest('$rir_cuts_manifest')
print(f'Successfully loaded {len(cuts)} RIR cuts')
print(f'Total duration: {cuts.total_duration():.2f} seconds')
print(f'Average duration: {cuts.total_duration()/len(cuts):.3f} seconds')
"
  
  log "RIR data preparation completed successfully!"
fi

log "To use RIR augmentation in training, add these options:"
log "  --enable-rir True"
log "  --rir-cuts-path $rir_cuts_manifest"
log "  --rir-prob 0.5  # Adjust probability as needed"
