#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir=
exp_dir=
world_size=1
num_epochs=1
max_duration=300
num_workers=4
use_fp16=true

. "${script_dir}/../../../icefall/shared/parse_options.sh" || exit 1

if [[ -z "${data_dir}" || -z "${exp_dir}" ]]; then
  echo "--data-dir and --exp-dir are required" >&2
  exit 2
fi

if [[ ! -f "${data_dir}/lang_phone/tokens.txt" ]]; then
  echo "Missing ${data_dir}/lang_phone/tokens.txt; run prepare.sh first" >&2
  exit 2
fi

for split in train dev; do
  if [[ ! -f "${data_dir}/fbank/cv-ja_cuts_${split}.jsonl.gz" ]]; then
    echo "Missing fbank cuts for ${split}; run prepare.sh through stage 2 first" >&2
    exit 2
  fi
done

mkdir -p "${exp_dir}"
started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
started_seconds="$(date +%s)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader > "${exp_dir}/gpu.csv"
fi

python "${script_dir}/zipformer/train.py" \
  --world-size "${world_size}" \
  --num-epochs "${num_epochs}" \
  --start-epoch 1 \
  --exp-dir "${exp_dir}" \
  --lang-dir "${data_dir}/lang_phone" \
  --language ja \
  --cv-manifest-dir "${data_dir}/fbank" \
  --manifest-dir "${data_dir}/fbank" \
  --enable-musan false \
  --num-workers "${num_workers}" \
  --max-duration "${max_duration}" \
  --use-fp16 "${use_fp16}" \
  --causal true \
  --chunk-size 16 \
  --left-context-frames 64 \
  --num-encoder-layers 1,1,1,1,1,1 \
  --feedforward-dim 192,192,192,192,192,192 \
  --encoder-dim 128,128,128,128,128,128 \
  --encoder-unmasked-dim 128,128,128,128,128,128 \
  --decoder-dim 320 \
  --joiner-dim 320

finished_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
elapsed_seconds="$(( $(date +%s) - started_seconds ))"
printf '{"started_at":"%s","finished_at":"%s","wall_seconds":%s,"num_epochs":%s,"world_size":%s}\n' \
  "${started_at}" "${finished_at}" "${elapsed_seconds}" "${num_epochs}" "${world_size}" \
  > "${exp_dir}/runtime.json"

echo "Training complete. Epoch timing: ${exp_dir}/runtime.json"
