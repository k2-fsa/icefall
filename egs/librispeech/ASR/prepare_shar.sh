#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail
set -x

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
#
#  - $dl_dir/lm
#      This directory contains the following files downloaded from
#       http://www.openslr.org/resources/11
#
#        - 3-gram.pruned.1e-7.arpa.gz
#        - 3-gram.pruned.1e-7.arpa
#        - 4-gram.arpa.gz
#        - 4-gram.arpa
#        - librispeech-vocab.txt
#        - librispeech-lexicon.txt
#        - librispeech-lm-norm.txt.gz
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Run data downloading and core manifest preparation
./prepare.sh --nj $nj --stage $stage --stop-stage 3

# Split the data into shards and compute the features on shard level
# This step leverages Lhotse Shar format for optimized sequential I/O
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: [Shar] Split manifests into shards and compute fbank features"
  mkdir -p data/shar
  if [ ! -e data/shar/.librispeech.done ]; then
     for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
       lhotse cut simple \
         -r data/manifests/librispeech_recordings_${part}.jsonl.gz \
         -s data/manifests/librispeech_supervisions_${part}.jsonl.gz \
         data/manifests/librispeech_cuts_${part}.jsonl.gz
     done

     lhotse combine \
       data/manifests/librispeech_cuts_train-{clean-100,clean-360,other-500}.jsonl.gz - \
       | shuf \
       | gzip -c \
       > data/manifests/librispeech_cuts_train_all.jsonl.gz

    lhotse shar export -j$nj -v -a flac -s 1000 \
      data/manifests/librispeech_cuts_train_all.jsonl.gz \
      data/shar

    lhotse shar compute-features -v -j$nj data/shar

    touch data/shar/.librispeech.done
  fi
fi

# Run the rest of data preparation steps
./prepare.sh --stage $stage --stop-stage $stop_stage
