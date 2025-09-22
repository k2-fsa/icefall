#!/usr/bin/env bash

# Copyright 2022 QCRI (author: Amir Hussein)
# Apache 2.0
# This script prepares the graphemic lexicon.

dir=data/local/dict
stage=0
lang_dir_src=$1
lang_dir_tgt=$2

cat $lang_dir_src/transcript_words.txt | tr -s " " "\n" | sort -u > $lang_dir_src/uniq_words
cat $lang_dir_tgt/transcript_words.txt | tr -s " " "\n" | sort -u > $lang_dir_tgt/uniq_words

echo "$0: processing lexicon text and creating lexicon... $(date)."
# remove vowels and  rare alef wasla
cat $lang_dir_src/uniq_words |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sed -r '/^\s*$/d' | sort -u > $lang_dir_src/words.txt
cat $lang_dir_tgt/uniq_words | sed -r '/^\s*$/d' | sort -u > $lang_dir_tgt/words.txt


echo "$0: Lexicon preparation succeeded"
