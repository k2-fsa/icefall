#!/bin/bash

# minimal_train.sh - 최소한의 설정으로 안정적인 훈련
set -euo pipefail

# 매우 보수적인 설정
world_size=1                    # 단일 GPU로 시작
max_duration=100                # 매우 작은 배치 크기
valid_max_duration=10          
num_buckets=50                 
num_workers=2                  
warm_step=10000
lang_dir="./data/lang_bpe_5000"
method="ctc-decoding"

# Model parameters
att_rate=0                    
num_decoder_layers=0          

# Other settings
start_epoch=19
master_port=12346
sanity_check=false           

# Validation 완전히 비활성화
enable_validation=false       

if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

echo "🚀 Starting minimal stable training..."
echo "World size: $world_size"
echo "Max duration: $max_duration"
echo "Validation: $enable_validation"

python3 ./conformer_ctc/train.py \
    --master-port $master_port \
    --sanity-check $sanity_check \
    --world-size $world_size \
    --warm-step $warm_step \
    --start-epoch $start_epoch \
    --att-rate $att_rate \
    --num-decoder-layers $num_decoder_layers \
    --num-workers $num_workers \
    --enable-spec-aug false \
    --enable-musan false \
    --enable-rir false \
    --rir-cuts-path data/rir/rir_cuts.jsonl.gz \
    --rir-prob 0.5 \
    --max-duration $max_duration \
    --valid-max-duration $valid_max_duration \
    --num-buckets $num_buckets \
    --bucketing-sampler true \
    --concatenate-cuts false \
    --duration-factor 1.0 \
    --drop-last true \
    --shuffle true \
    --lang-dir $lang_dir \
    --method $method \
    --enable-validation $enable_validation
