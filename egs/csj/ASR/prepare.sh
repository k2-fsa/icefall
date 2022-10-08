#!/usr/bin/env bash

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume the following directories are downloaded.
#
#  - $csj_dir
#  - $musan_dir
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#     - music
#     - noise
#     - speech

csj_dir=/mnt/minami_data_server/t2131178/corpus/CSJ
musan_dir=/mnt/minami_data_server/t2131178/corpus/musan/musan
trans_dir=$csj_dir/retranscript_new
fbank_dir=$csj_dir/fbank_new
csj_manifest_dir=data/manifests
# exp_dir=conv_emformer_transducer_stateless2/exp_espnet

. shared/parse_options.sh || exit 1

mkdir -p data

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then 
    log "Stage 0: Make CSJ Transcript"
    python -O local/csj_make_transcript.py --corpus-dir $csj_dir \
        --trans-dir $trans_dir --config local/conf/disfluent.ini --write-segments

    modes=(
        symbol
        fluent
        number
    )
    for mode in ${modes[@]}; do
        python -O local/csj_make_transcript.py --corpus-dir $csj_dir \
            --trans-dir $trans_dir --config local/conf/$mode.ini --use-segments
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then 
    log "Stage 1: Prepare CSJ manifest"
    python local/lhotse_prepare_csj.py --trans-dir $trans_dir \
        --manifest-dir data/manifests --split 4000
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Prepare musan manifest"
    mkdir -p $musan_dir/manifests
    if [ ! -e $musan_dir/manifests/.musan.done ]; then
        lhotse prepare musan $musan_dir $musan_dir/manifests
        touch $musan_dir/manifests/.musan.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then 
    log "Stage 3: Prepare CSJ lang"
    modes=(
        disfluent 
        symbol 
        fluent 
        number
    )
    for mode in ${modes[@]}; do
        python local/prepare_lang_char.py --trans-mode $mode \
            --train-cuts data/manifests/cuts_train.jsonl.gz \
            --lang-dir lang_char_$mode
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Compute CSJ fbank"
    python local/compute_fbank_csj.py --manifest-dir data/manifests \
        --fbank-dir $fbank_dir
    parts=(
        train 
        valid
        eval1
        eval2
        eval3
    )
    for part in ${parts[@]}; do 
        python local/validate_manifest.py --manifest data/manifests/cuts_$part.jsonl.gz
    done
    touch $fbank_dir/.csj-validated.done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank for musan"
    mkdir -p $musan_dir/fbank

    if [ ! -e $musan_dir/fbank/.musan.done ]; then 
        python -O local/compute_fbank_musan.py --manifest-dir $musan_dir/manifests --fbank-dir $musan_dir/fbank
        touch $musan_dir/fbank/.musan.done
    fi
fi