#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

# This script generates Ngram LM and related files needed by decoding.

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/lm
#      This directory contains the language model downloaded from
#        https://huggingface.co/wgb14/gigaspeech_lm
#
#        - 3gram_pruned_1e7.arpa.gz
#        - 4gram.arpa.gz
#        - lexicon.txt

. prepare.sh --stage -1 --stop-stage 9 || exit 1

stage=0
stop_stage=100

. shared/parse_options.sh || exit 1

log "Running prepare_lm.sh"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare BPE based lexicon"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone/words.txt $lang_dir

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir
  
      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare word-level G"
  # We assume you have installed kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm

  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3gram_pruned_1e7.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $dl_dir/lm/4gram.arpa > data/lm/G_4_gram.fst.txt
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compile LG"
  # It is used for for RNN-T fast_beam_search decoding
  ./local/compile_lg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_lg.py --lang-dir $lang_dir
  done
fi
