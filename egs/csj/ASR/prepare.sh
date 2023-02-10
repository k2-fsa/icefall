#!/usr/bin/env bash
# We assume the following directories are downloaded.
#
#  - $csj_dir
#     CSJ is assumed to be the USB-type directory, which should contain the following subdirectories:-
#     - DATA (not used in this script)
#     - DOC (not used in this script)
#     - MODEL (not used in this script)
#     - MORPH
#       - LDB (not used in this script)
#       - SUWDIC (not used in this script)
#       - SDB
#         - core
#           - ...
#         - noncore
#           - ...
#     - PLABEL (not used in this script)
#     - SUMMARY (not used in this script)
#     - TOOL (not used in this script)
#     - WAV
#       - core
#         - ...
#       - noncore
#         - ...
#     - XML (not used in this script)
#
#  - $musan_dir
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#     - music
#     - noise
#     - speech
#
# By default, this script produces the original transcript like kaldi and espnet. Optionally, you
# can generate other transcript formats by supplying your own config files. A few examples of these
# config files can be found in local/conf.

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=8
stage=-1
stop_stage=100

csj_dir=/mnt/minami_data_server/t2131178/corpus/CSJ
musan_dir=/mnt/minami_data_server/t2131178/corpus/musan/musan
trans_dir=$csj_dir/retranscript
csj_fbank_dir=/mnt/host/csj_data/fbank
musan_fbank_dir=$musan_dir/fbank
csj_manifest_dir=data/manifests
musan_manifest_dir=$musan_dir/manifests

. shared/parse_options.sh || exit 1

mkdir -p data

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare CSJ manifest"
    # If you want to generate more transcript modes, append the path to those config files at c.
    # Example: lhotse prepare csj $csj_dir $trans_dir $csj_manifest_dir -c local/conf/disfluent.ini
    # NOTE: In case multiple config files are supplied, the second config file and onwards will inherit
    #       the segment boundaries of the first config file.
    if [ ! -e $csj_manifest_dir/.csj.done ]; then
        lhotse prepare csj $csj_dir $trans_dir $csj_manifest_dir -j 4
        touch $csj_manifest_dir/.csj.done
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Prepare musan manifest"
    mkdir -p $musan_manifest_dir
    if [ ! -e $musan_manifest_dir/.musan.done ]; then
        lhotse prepare musan $musan_dir $musan_manifest_dir
        touch $musan_manifest_dir/.musan.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Compute CSJ fbank"
    if [ ! -e $csj_fbank_dir/.csj-validated.done ]; then
        python local/compute_fbank_csj.py --manifest-dir $csj_manifest_dir \
            --fbank-dir $csj_fbank_dir
        parts=(
            train
            valid
            eval1
            eval2
            eval3
        )
        for part in ${parts[@]}; do
            python local/validate_manifest.py --manifest $csj_manifest_dir/csj_cuts_$part.jsonl.gz
        done
        touch $csj_fbank_dir/.csj-validated.done
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Prepare CSJ lang"
    modes=disfluent

    # If you want prepare the lang directory for other transcript modes, just append
    # the names of those modes behind. An example is shown as below:-
    # modes="$modes fluent symbol number"

    for mode in ${modes[@]}; do
        python local/prepare_lang_char.py --trans-mode $mode \
            --train-cut $csj_manifest_dir/csj_cuts_train.jsonl.gz \
            --lang-dir lang_char_$mode
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank for musan"
    mkdir -p $musan_fbank_dir

    if [ ! -e $musan_fbank_dir/.musan.done ]; then
        python local/compute_fbank_musan.py --manifest-dir $musan_manifest_dir --fbank-dir $musan_fbank_dir
        touch $musan_fbank_dir/.musan.done
    fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Show manifest statistics"
    python local/display_manifest_statistics.py --manifest-dir $csj_manifest_dir > $csj_manifest_dir/manifest_statistics.txt
    cat $csj_manifest_dir/manifest_statistics.txt
fi
