#!/usr/bin/env bash

# Copyright 2022 QCRI (author: Amir Hussein)
# Apache 2.0
# This script prepares the graphemic lexicon.

dir=data/local/dict
stage=0
lang_dir=$1

cat $lang_dir/transcript_words.txt | tr -s " " "\n" | sort -u > $lang_dir/uniq_words

echo "$0: processing lexicon text and creating lexicon... $(date)."
# remove vowels and  rare alef wasla
cat $lang_dir/uniq_words |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sed -r '/^\s*$/d' | sort -u > $lang_dir/words.txt


echo "$0: Lexicon preparation succeeded"
