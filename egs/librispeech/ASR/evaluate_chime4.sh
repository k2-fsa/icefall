#!/bin/bash
# CHiME-4 evaluation script for conformer_ctc

set -euo pipefail

# Configuration
CHECKPOINT_PATH="conformer_ctc/exp/pretrained.pt"  # Update with your actual checkpoint
LOG_LEVEL="INFO"
DEVICE="cuda"

echo "CHiME-4 Evaluation for Conformer CTC"
echo "====================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please train a model first or specify correct checkpoint path"
    exit 1
fi

# Check if CHiME-4 data exists
if [ ! -d "/home/nas/DB/CHiME4/data/audio/16kHz/isolated" ]; then
    echo "Error: CHiME-4 data not found at /home/nas/DB/CHiME4/data/audio/16kHz/isolated"
    echo "Please check CHiME-4 data path"
    exit 1
fi

echo "Starting CHiME-4 evaluation..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Device: $DEVICE"
echo ""

# Run evaluation
python evaluate_chime4.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --manifest-dir data/fbank \
    --max-duration 200.0 \
    --log-level "$LOG_LEVEL" \
    --device "$DEVICE"

echo ""
echo "CHiME-4 evaluation completed!"
