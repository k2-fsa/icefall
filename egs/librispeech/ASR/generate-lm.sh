#!/usr/bin/env bash

lang_dir=data/lang_bpe_500
if [ ! -f $lang_dir/bigram.arpa ]; then
  ./shared/make_kn_lm.py \
    -ngram-order 2 \
    -text $lang_dir/transcript_tokens.txt \
    -lm $lang_dir/bigram.arpa
fi

if [ ! -f $lang_dir/bigram.fst.txt ]; then
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/tokens.txt" \
    --disambig-symbol='#0' \
    --max-order=2 \
    $lang_dir/bigram.arpa > $lang_dir/bigram.fst.txt
fi

if [ ! -f $lang_dir/trigram.arpa ]; then
  ./shared/make_kn_lm.py \
    -ngram-order 3 \
    -text $lang_dir/transcript_tokens.txt \
    -lm $lang_dir/trigram.arpa
fi

if [ ! -f $lang_dir/trigram.fst.txt ]; then
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/tokens.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    $lang_dir/trigram.arpa > $lang_dir/trigram.fst.txt
fi
