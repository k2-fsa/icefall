#!/usr/bin/env bash

lang_dir=data/lang_bpe_500
if [ ! -f $lang_dir/2gram.arpa ]; then
  ./shared/make_kn_lm.py \
    -ngram-order 2 \
    -text $lang_dir/transcript_tokens.txt \
    -lm $lang_dir/2gram.arpa
fi

if [ ! -f $lang_dir/2gram.fst.txt ]; then
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/tokens.txt" \
    --disambig-symbol='#0' \
    --max-order=2 \
    $lang_dir/2gram.arpa > $lang_dir/2gram.fst.txt
fi

if [ ! -f $lang_dir/3gram.arpa ]; then
  ./shared/make_kn_lm.py \
    -ngram-order 3 \
    -text $lang_dir/transcript_tokens.txt \
    -lm $lang_dir/3gram.arpa
fi

if [ ! -f $lang_dir/3gram.fst.txt ]; then
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/tokens.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    $lang_dir/3gram.arpa > $lang_dir/3gram.fst.txt
fi


if [ ! -f $lang_dir/5gram.arpa ]; then
  ./shared/make_kn_lm.py \
    -ngram-order 5 \
    -text $lang_dir/transcript_tokens.txt \
    -lm $lang_dir/5gram.arpa
fi

if [ ! -f $lang_dir/5gram.fst.txt ]; then
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/tokens.txt" \
    --disambig-symbol='#0' \
    --max-order=5 \
    $lang_dir/5gram.arpa > $lang_dir/5gram.fst.txt
fi
