#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

commonvoice_root=
data_dir="${script_dir}/data"
stage=0
stop_stage=100
nj=8
batch_duration=200

. "${script_dir}/../../../icefall/shared/parse_options.sh" || exit 1

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}) $*"
}

if [[ -z "${commonvoice_root}" ]]; then
  echo "--commonvoice-root must point directly to the Japanese Common Voice directory" >&2
  exit 2
fi

if [[ ! -d "${commonvoice_root}/clips" ]]; then
  echo "Missing clips/ under --commonvoice-root: ${commonvoice_root}" >&2
  exit 2
fi

for split in train dev test; do
  if [[ ! -f "${commonvoice_root}/${split}.tsv" ]]; then
    echo "Missing ${split}.tsv under --commonvoice-root: ${commonvoice_root}" >&2
    exit 2
  fi
done

mkdir -p "${data_dir}"

if [[ ${stage} -le 0 && ${stop_stage} -ge 0 ]]; then
  log "Stage 0: Prepare Japanese Common Voice manifests and normalized cuts"
  source_parent="${data_dir}/.commonvoice-source"
  mkdir -p "${source_parent}" "${data_dir}/manifests"
  ln -sfn "${commonvoice_root}" "${source_parent}/ja"

  if [[ ! -f "${data_dir}/manifests/.cv-ja.done" ]]; then
    python "${script_dir}/local/lhotse_cli.py" prepare commonvoice --language ja -j "${nj}" \
      "${source_parent}" "${data_dir}/manifests"
    touch "${data_dir}/manifests/.cv-ja.done"
  fi

  python "${script_dir}/local/prepare_commonvoice.py" \
    --manifest-dir "${data_dir}/manifests" \
    --output-dir "${data_dir}/fbank"
fi

if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
  log "Stage 1: Build the phone-token inventory from training transcripts"
  python "${script_dir}/local/prepare_tokens.py" \
    --cuts "${data_dir}/fbank/cv-ja_cuts_train_raw.jsonl.gz" \
    --lang-dir "${data_dir}/lang_phone"
fi

if [[ ${stage} -le 2 && ${stop_stage} -ge 2 ]]; then
  log "Stage 2: Compute fbank features"
  python "${script_dir}/local/compute_fbank.py" \
    --data-dir "${data_dir}" \
    --num-workers "${nj}" \
    --batch-duration "${batch_duration}"
fi

log "Preparation complete: ${data_dir}"
