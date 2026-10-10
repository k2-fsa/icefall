#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir=
exp_dir=
checkpoint=
split=dev
epoch=30
avg=15
use_averaged_model=true
decoding_method=greedy_search
blank_penalty=0.0
beam_size=4
max_duration=600
num_workers=4

. "${script_dir}/../../../icefall/shared/parse_options.sh" || exit 1

if [[ -z "${data_dir}" || -z "${exp_dir}" ]]; then
  echo "--data-dir and --exp-dir are required" >&2
  exit 2
fi
if [[ "${split}" != "dev" && "${split}" != "test" ]]; then
  echo "--split must be dev or test" >&2
  exit 2
fi
if ((avg < 1 || epoch < 1)); then
  echo "--epoch and --avg must be positive" >&2
  exit 2
fi

required_files=(
  "${data_dir}/lang_phone/tokens.txt"
  "${data_dir}/fbank/cv-ja_cuts_${split}.jsonl.gz"
)
model_args=()
if [[ -n "${checkpoint}" ]]; then
  required_files+=("${exp_dir}/${checkpoint}")
  model_args+=(--checkpoint "${checkpoint}")
elif [[ "${use_averaged_model}" == "true" ]]; then
  required_files+=(
    "${exp_dir}/epoch-$((epoch - avg)).pt"
    "${exp_dir}/epoch-${epoch}.pt"
  )
  model_args+=(
    --epoch "${epoch}"
    --avg "${avg}"
    --use-averaged-model true
  )
else
  for checkpoint_epoch in $(seq "$((epoch - avg + 1))" "${epoch}"); do
    required_files+=("${exp_dir}/epoch-${checkpoint_epoch}.pt")
  done
  model_args+=(
    --epoch "${epoch}"
    --avg "${avg}"
    --use-averaged-model false
  )
fi

for required in "${required_files[@]}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required evaluation input: ${required}" >&2
    exit 2
  fi
done

python "${script_dir}/zipformer/decode.py" \
  "${model_args[@]}" \
  --split "${split}" \
  --decoding-method "${decoding_method}" \
  --blank-penalty "${blank_penalty}" \
  --beam-size "${beam_size}" \
  --exp-dir "${exp_dir}" \
  --output-dir "${exp_dir}/per" \
  --lang-dir "${data_dir}/lang_phone" \
  --language ja \
  --cv-manifest-dir "${data_dir}/fbank" \
  --manifest-dir "${data_dir}/fbank" \
  --enable-musan false \
  --shuffle false \
  --drop-last false \
  --num-workers "${num_workers}" \
  --max-duration "${max_duration}" \
  --causal true \
  --chunk-size 16 \
  --left-context-frames 64 \
  --num-encoder-layers 1,1,1,1,1,1 \
  --feedforward-dim 192,192,192,192,192,192 \
  --encoder-dim 128,128,128,128,128,128 \
  --encoder-unmasked-dim 128,128,128,128,128,128 \
  --decoder-dim 320 \
  --joiner-dim 320 \
  "$@"
