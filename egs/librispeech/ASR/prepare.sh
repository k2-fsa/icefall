#!/usr/bin/env bash

set -eou pipefail

nj=15
stage=-1
stop_stage=100

. local/parse_options.sh || exit 1

mkdir -p data

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  echo "stage -1: Download LM"
  mkdir -p data/lm
  ./local/download_lm.py
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriSpeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriSpeech data/
  #
  # The script checks that if
  #
  #   data/LibriSpeech/test-clean/.completed exists,
  #
  # it will not re-download it.
  #
  # The same goes for dev-clean, dev-other, test-other, train-clean-100
  # train-clean-360, and train-other-500

  mkdir -p data/LibriSpeech
  lhotse download librispeech --full data

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan data/
  #
  # and create a file data/.musan_completed
  # to avoid downloading it again
  if [ ! -f data/.musan_completed ]; then
    lhotse download musan data
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare librispeech manifest"
  # We assume that you have downloaded the librispeech corpus
  # to data/LibriSpeech
  mkdir -p data/manifests
  lhotse prepare librispeech -j $nj data/LibriSpeech data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  lhotse prepare musan data/musan data/manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Compute fbank for librispeech"
  mkdir -p data/fbank
  ./local/compute_fbank_librispeech.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: Prepare phone based lang"
  # TODO: add BPE based lang
  mkdir -p data/lang

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - data/lm/librispeech-lexicon.txt |
    sort | uniq > data/lang/lexicon.txt

  ./local/prepare_lang.py
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "Stage 6: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  if [ ! -e data/lm/G_3_gram.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/lm/3-gram.pruned.1e-7.arpa > data/lm/G_3_gram.fst.txt
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "Stage 7: Compile HLG"
  if [ ! -f data/lm/HLG.pt ]; then
    python3 ./local/compile_hlg.py
  fi
fi
