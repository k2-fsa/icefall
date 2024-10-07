#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

# This script generate Ngram LM / NNLM and related files that needed by decoding.

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
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

. prepare.sh --stage -1 --stop-stage 6 || exit 1

log "Running prepare_lm.sh"

stage=0
stop_stage=100

. shared/parse_options.sh || exit 1

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare BPE based lexicon."

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
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

    if [ ! -f $lang_dir/L.fst ]; then
      log "Converting L.pt to L.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L.pt \
        $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
      log "Converting L_disambig.pt to L_disambig.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L_disambig.pt \
        $lang_dir/L_disambig.fst
    fi
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare word level G"
  # We assume you have installed kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3-gram.pruned.1e-7.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $dl_dir/lm/4-gram.arpa > data/lm/G_4_gram.fst.txt
  fi

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/HL.fst ]; then
      ./local/prepare_lang_fst.py  \
        --lang-dir $lang_dir \
        --ngram-G ./data/lm/G_3_gram.fst.txt
    fi
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone

  # Note If ./local/compile_hlg.py throws OOM,
  # please switch to the following command
  #
  # ./local/compile_hlg_using_openfst.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir

    # Note If ./local/compile_hlg.py throws OOM,
    # please switch to the following command
    #
    # ./local/compile_hlg_using_openfst.py --lang-dir $lang_dir
  done
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compile LG"
  ./local/compile_lg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_lg.py --lang-dir $lang_dir
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare token level ngram G"
  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/transcript_tokens.txt ]; then
      ./local/convert_transcript_words_to_tokens.py \
        --lexicon $lang_dir/lexicon.txt \
        --transcript $lang_dir/transcript_words.txt \
        --oov "<UNK>" \
        > $lang_dir/transcript_tokens.txt
    fi

    for ngram in 2 3 4 5; do
      if [ ! -f $lang_dir/${ngram}gram.arpa ]; then
        ./shared/make_kn_lm.py \
          -ngram-order ${ngram} \
          -text $lang_dir/transcript_tokens.txt \
          -lm $lang_dir/${ngram}gram.arpa
      fi
    
      if [ ! -f $lang_dir/${ngram}gram.fst.txt ]; then
        python3 -m kaldilm \
          --read-symbol-table="$lang_dir/tokens.txt" \
          --disambig-symbol='#0' \
          --max-order=${ngram} \
          $lang_dir/${ngram}gram.arpa > $lang_dir/${ngram}gram.fst.txt
      fi
    done
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Generate NNLM training data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    lang_dir=data/lang_bpe_${vocab_size}
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $dl_dir/lm/librispeech-lm-norm.txt \
      --lm-archive $out_dir/lm_data.pt
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Generate NNLM validation data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    if [ ! -f $out_dir/valid.txt ]; then
      gunzip -c data/manifests/libritts_supervisions_dev-clean.jsonl.gz \
        | jq ".text" | sed 's/"//g' \
        | ./local/norm_text.py > $out_dir/valid.txt

      gunzip -c data/manifests/libritts_supervisions_dev-other.jsonl.gz \
        | jq ".text" | sed 's/"//g' \
        | ./local/norm_text.py >> $out_dir/valid.txt
    fi

    lang_dir=data/lang_bpe_${vocab_size}
    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $out_dir/valid.txt \
      --lm-archive $out_dir/lm_data-valid.pt
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Generate NNLM test data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    if [ ! -f $out_dir/test.txt ]; then
      gunzip -c data/manifests/libritts_supervisions_test-clean.jsonl.gz \
        | jq ".text" | sed 's/"//g' \
        | ./local/norm_text.py > $out_dir/test.txt

      gunzip -c data/manifests/libritts_supervisions_test-other.jsonl.gz \
        | jq ".text" | sed 's/"//g' \
        | ./local/norm_text.py >> $out_dir/test.txt
    fi

    lang_dir=data/lang_bpe_${vocab_size}
    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $out_dir/test.txt \
      --lm-archive $out_dir/lm_data-test.pt
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Sort NNLM training data"
  # Sort LM training data by sentence length in descending order
  # for ease of training.
  #
  # Sentence length equals to the number of BPE tokens
  # in a sentence.

  for vocab_size in ${vocab_sizes[@]}; do
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir
    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data.pt \
      --out-lm-data $out_dir/sorted_lm_data.pt \
      --out-statistics $out_dir/statistics.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-valid.pt \
      --out-lm-data $out_dir/sorted_lm_data-valid.pt \
      --out-statistics $out_dir/statistics-valid.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-test.pt \
      --out-lm-data $out_dir/sorted_lm_data-test.pt \
      --out-statistics $out_dir/statistics-test.txt
  done
fi
