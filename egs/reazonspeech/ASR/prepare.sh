#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=8
stage=-1
stop_stage=100

reazonspeech_dir=corpus
reazonspeech_manifest_dir=data

. shared/parse_options.sh || exit 1

mkdir -p data

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare ReazonSpeech manifest"
    if [ ! -e $reazonspeech_manifest_dir/.reazonspeech.done ]; then
        lhotse prepare reazonspeech $reazonspeech_dir $reazonspeech_manifest_dir
        touch $reazonspeech_manifest_dir/.reazonspeech.done
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Compute ReazonSpeech fbank"
    if [ ! -e $reazonspeech_manifest_dir/.reazonspeech-validated.done ]; then
        python local/compute_fbank_reazonspeech.py --manifest-dir $reazonspeech_manifest_dir
        python local/validate_manifest.py --manifest $reazonspeech_manifest_dir/reazonspeech_cuts_train.jsonl.gz
        python local/validate_manifest.py --manifest $reazonspeech_manifest_dir/reazonspeech_cuts_valid.jsonl.gz
        python local/validate_manifest.py --manifest $reazonspeech_manifest_dir/reazonspeech_cuts_test.jsonl.gz
        touch $reazonspeech_manifest_dir/.reazonspeech-validated.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare ReazonSpeech lang_char"
    python local/prepare_lang_char.py $reazonspeech_manifest_dir/reazonspeech_cuts_train.jsonl.gz
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Show manifest statistics"
    python local/display_manifest_statistics.py --manifest-dir $reazonspeech_manifest_dir > $reazonspeech_manifest_dir/manifest_statistics.txt
    cat $reazonspeech_manifest_dir/manifest_statistics.txt
fi
