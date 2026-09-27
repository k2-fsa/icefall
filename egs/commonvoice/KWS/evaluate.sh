#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir=
exp_dir=
checkpoint=best-valid-loss.pt
decoding_method=greedy_search
max_duration=600
num_workers=4

. "${script_dir}/../../../icefall/shared/parse_options.sh" || exit 1

if [[ -z "${data_dir}" || -z "${exp_dir}" ]]; then
  echo "--data-dir and --exp-dir are required" >&2
  exit 2
fi

for required in \
  "${data_dir}/lang_phone/tokens.txt" \
  "${data_dir}/fbank/cv-ja_cuts_test.jsonl.gz" \
  "${exp_dir}/${checkpoint}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required evaluation input: ${required}" >&2
    exit 2
  fi
done

python "${script_dir}/zipformer/decode.py" \
  --checkpoint "${checkpoint}" \
  --decoding-method "${decoding_method}" \
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
