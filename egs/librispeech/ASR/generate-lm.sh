#!/usr/bin/env bash

lang_dir=data/lang_bpe_500

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
