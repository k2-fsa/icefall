#!/usr/bin/env bash

set -eou pipefail

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
#
#  - $do_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=$PWD/download

. shared/parse_options.sh || exit 1


# All generated files by this script are saved in "data"
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "stage -1: Download LM"
  ./local/download_lm.py --out-dir=$dl_dir/lm
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriSpeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriSpeech $dl_dir/LibriSpeech
  #
  if [ ! -d $dl_dir/LibriSpeech/train-other-500 ]; then
    lhotse download librispeech --full $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriSpeech manifest"
  # We assume that you have downloaded the LibriSpeech corpus
  # to $dl_dir/LibriSpeech
  mkdir -p data/manifests
  lhotse prepare librispeech -j $nj $dl_dir/LibriSpeech data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for librispeech"
  mkdir -p data/fbank
  ./local/compute_fbank_librispeech.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  mkdir -p data/lang_phone

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/librispeech-lexicon.txt |
    sort | uniq > data/lang_phone/lexicon.txt

  if [ ! -f data/lang_phone/L_disambig.pt ]; then
    ./local/prepare_lang.py
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "State 6: Prepare BPE based lang"
  mkdir -p data/lang_bpe
  # We reuse words.txt from phone based lexicon
  # so that the two can share G.pt later.
  cp data/lang_phone/words.txt data/lang_bpe/

  if [ ! -f data/lang_bpe/train.txt ]; then
    log "Generate data for BPE training"
    files=$(
      find "data/LibriSpeech/train-clean-100" -name "*.trans.txt"
      find "data/LibriSpeech/train-clean-360" -name "*.trans.txt"
      find "data/LibriSpeech/train-other-500" -name "*.trans.txt"
    )
    for f in ${files[@]}; do
      cat $f | cut -d " " -f 2-
    done > data/lang_bpe/train.txt
  fi

  python3 ./local/train_bpe_model.py

  if [ ! -f data/lang_bpe/L_disambig.pt ]; then
    ./local/prepare_lang_bpe.py
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare bigram P"
  if [ ! -f data/lang_bpe/corpus.txt ]; then
    ./local/convert_transcript_to_corpus.py \
      --lexicon data/lang_bpe/lexicon.txt \
      --transcript data/lang_bpe/train.txt \
      --oov "<UNK>" \
      > data/lang_bpe/corpus.txt
  fi

  if [ ! -f data/lang_bpe/P.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 2 \
      -text data/lang_bpe/corpus.txt \
      -lm data/lang_bpe/P.arpa
  fi

  # TODO: Use egs/wsj/s5/utils/lang/ngram_entropy_pruning.py
  # from kaldi to prune P if it causes OOM later

  if [ ! -f data/lang_bpe/P-no-prune.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_bpe/tokens.txt" \
      --disambig-symbol='#0' \
      --max-order=2 \
      data/lang_bpe/P.arpa > data/lang_bpe/P-no-prune.fst.txt
  fi

  thresholds=(
    1e-6
    1e-7
  )
  for threshold in ${thresholds[@]}; do
    if [ ! -f data/lang_bpe/P-pruned.${threshold}.arpa ]; then
      python3 ./local/ngram_entropy_pruning.py \
        -threshold $threshold \
        -lm data/lang_bpe/P.arpa \
        -write-lm data/lang_bpe/P-pruned.${threshold}.arpa
    fi

    if [ ! -f data/lang_bpe/P-pruned.${threshold}.fst.txt ]; then
      python3 -m kaldilm \
        --read-symbol-table="data/lang_bpe/tokens.txt" \
        --disambig-symbol='#0' \
        --max-order=2 \
        data/lang_bpe/P-pruned.${threshold}.arpa > data/lang_bpe/P-pruned.${threshold}.fst.txt
    fi
  done

  if [ ! -f data/lang_bpe/P-uni.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_bpe/tokens.txt" \
      --disambig-symbol='#0' \
      --max-order=1 \
      data/lang_bpe/P.arpa > data/lang_bpe/P-uni.fst.txt
  fi

  ( cd data/lang_bpe;
    # ln -sfv P-pruned.1e-6.fst.txt P.fst.txt
    ln -sfv P-no-prune.fst.txt P.fst.txt
  )
  rm -fv data/lang_bpe/P.pt data/lang_bpe/ctc_topo_P.pt
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"
  # We assume you have install kaldilm, if not, please install
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
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"
  python3 ./local/compile_hlg.py
fi
