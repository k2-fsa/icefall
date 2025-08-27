#!/bin/# Data Augmentation Controls (modify these as needed)
enable_spec_aug=true          # SpecAugment (frequency/time masking)
enable_musan=true             # MUSAN noise augmentation
enable_rir=true              # RIR (Room Impulse Response) augmentation - FIXED AND RE-ENABLED
enable_cutmix=true           # Cut mixing: 두 오디오의 시간 구간을 섞음
enable_concatenate=true       # Cut concatenation: 짧은 발화들을 연결하여 패딩 최소화
# train.sh - LibriSpeech ASR Training Script with Data Augmentation Control
# Usage: bash train.sh

set -euo pipefail

# Data Augmentation Controls (modify these as needed)
enable_spec_aug=true          # SpecAugment (frequency/time masking)
enable_musan=true             # MUSAN noise augmentation
enable_rir=true              # RIR (Room Impulse Response) augmentation - RE-ENABLED
enable_cutmix=true           # Cut mixing: 두 오디오의 시간 구간을 섞음
enable_concatenate=true       # Cut concatenation: 짧은 발화들을 연결하여 패딩 최소화

# RIR settings (used when enable_rir=true)
rir_cuts_path="data/manifests/rir.scp"  # Path to RIR file list (updated to use rir.scp)
rir_prob=0.5                  # Probability of applying RIR

# Training parameters
world_size=4                    # Multi-GPU restored since test passed
max_duration=300                # Further reduced from 320 to save memory
valid_max_duration=15          # Very small for multi-GPU safety  
num_buckets=200                 # Reduced for memory saving
num_workers=24                  # Much smaller to save memory
warm_step=40000
lang_dir="./data/lang_bpe_5000"
method="ctc-decoding"

# Model parameters
att_rate=0                    # 0 for pure CTC, >0 for CTC+Attention
num_decoder_layers=0          # 0 for pure CTC

# Other settings
start_epoch=0
master_port=12345
sanity_check=false           # Set to true for OOM checking (slower)

# Validation settings
enable_validation=true       # Set to false to disable validation completely  
valid_interval=5000           # Increased from 50 to allow more training before validation

# Validation decoding settings
validation_decoding_method="greedy"    # "greedy" or "beam" - use greedy for faster validation
validation_search_beam=10.0            # Beam size for validation (only used if method="beam")
validation_output_beam=5.0             # Output beam for validation (only used if method="beam")
validation_skip_wer=false              # Skip WER computation for even faster validation (디버깅용 - 이제 false로 변경)

if [ "$enable_rir" = "true" ]; then
    echo "  - RIR Path: $rir_cuts_path"
    echo "  - RIR Probability: $rir_prob"
fi


# gdb --args python ./conformer_ctc/train.py
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi


python3 ./conformer_ctc/train.py \
    --master-port $master_port \
    --sanity-check $sanity_check \
    --world-size $world_size \
    --warm-step $warm_step \
    --start-epoch $start_epoch \
    --att-rate $att_rate \
    --num-decoder-layers $num_decoder_layers \
    --num-workers $num_workers \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --enable-rir $enable_rir \
    --rir-cuts-path $rir_cuts_path \
    --rir-prob $rir_prob \
    --on-the-fly-feats true \
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --bucketing-sampler true \
    --concatenate-cuts $enable_concatenate \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle true \
    --lang-dir $lang_dir \
    --method $method \
    --enable-validation $enable_validation \
    --valid-interval $valid_interval \
    --validation-decoding-method $validation_decoding_method \
    --validation-search-beam $validation_search_beam \
    --validation-output-beam $validation_output_beam \
    --validation-skip-wer $validation_skip_wer
