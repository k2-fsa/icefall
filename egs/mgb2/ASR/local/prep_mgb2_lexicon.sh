#!/usr/bin/env bash

# Copyright 2022 QCRI (author: Amir Hussein)
# Apache 2.0
# This script prepares the graphemic lexicon.

dir=data/local/dict
lexicon_url1="https://arabicspeech.org/arabicspeech-portal-resources/lexicon/ar-ar_grapheme_lexicon_20160209.bz2";
lexicon_url2="https://arabicspeech.org/arabicspeech-portal-resources/lexicon/ar-ar_phoneme_lexicon_20140317.bz2";
stage=0
lang_dir=download/lm
mkdir -p $lang_dir

if [ $stage -le 0 ]; then
  echo "$0: Downloading text for lexicon... $(date)."
  wget --no-check-certificate -P $lang_dir $lexicon_url1
  wget --no-check-certificate -P $lang_dir $lexicon_url2
  bzcat $lang_dir/ar-ar_grapheme_lexicon_20160209.bz2 | sed '1,3d' | awk '{print $1}'  >  $lang_dir/grapheme_lexicon
  bzcat $lang_dir/ar-ar_phoneme_lexicon_20140317.bz2 | sed '1,3d' | awk '{print $1}' >>  $lang_dir/phoneme_lexicon
  cat download/lm/train/text | cut -d ' ' -f 2- | tr -s " " "\n" | sort -u >> $lang_dir/uniq_words
fi


if [ $stage -le 0 ]; then
  echo "$0: processing lexicon text and creating lexicon... $(date)."
  # remove vowels and  rare alef wasla
  cat $lang_dir/uniq_words |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sed -r '/^\s*$/d' | sort -u > $lang_dir/grapheme_lexicon.txt
fi

echo "$0: Lexicon preparation succeeded"
