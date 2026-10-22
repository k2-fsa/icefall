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
commonvoice_root="$(cd "${commonvoice_root}" && pwd -P)"
source_parent="${data_dir}/.commonvoice-source"
source_link="${source_parent}/ja"
source_record="${data_dir}/.commonvoice-source-root"

if [[ -f "${source_record}" ]]; then
  recorded_root="$(<"${source_record}")"
  if [[ "${recorded_root}" != "${commonvoice_root}" ]]; then
    echo "${data_dir} was prepared from ${recorded_root}; use a new --data-dir for ${commonvoice_root}" >&2
    exit 2
  fi
elif [[ -L "${source_link}" ]]; then
  if [[ ! -d "${source_link}" ]]; then
    echo "Cannot verify the existing Common Voice source link: ${source_link}" >&2
    exit 2
  fi
  recorded_root="$(cd "${source_link}" && pwd -P)"
  if [[ "${recorded_root}" != "${commonvoice_root}" ]]; then
    echo "${data_dir} was prepared from ${recorded_root}; use a new --data-dir for ${commonvoice_root}" >&2
    exit 2
  fi
  printf '%s\n' "${commonvoice_root}" > "${source_record}"
elif [[ -e "${data_dir}/manifests/.cv-ja.done" || -d "${data_dir}/fbank" ]]; then
  echo "Cannot verify the source of existing data in ${data_dir}; use a new --data-dir" >&2
  exit 2
else
  printf '%s\n' "${commonvoice_root}" > "${source_record}"
fi

if [[ ${stage} -le 0 && ${stop_stage} -ge 0 ]]; then
  if ! command -v lhotse >/dev/null 2>&1; then
    echo "Missing lhotse CLI; install Lhotse with its console script before preparation" >&2
    exit 2
  fi
  log "Stage 0: Prepare Japanese Common Voice manifests and normalized cuts"
  mkdir -p "${source_parent}" "${data_dir}/manifests"
  ln -sfn "${commonvoice_root}" "${source_link}"

  if [[ ! -f "${data_dir}/manifests/.cv-ja.done" ]]; then
    lhotse prepare commonvoice --language ja -j "${nj}" \
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
