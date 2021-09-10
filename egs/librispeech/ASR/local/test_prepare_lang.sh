#!/usr/bin/env bash

lang_dir=tmp_lang
mkdir -p $lang_dir
cat <<EOF > $lang_dir/lexicon.txt
<UNK> SPN
f f
a a
foo f o o
bar b a r
bark b a r k
food f o o d
food2 f o o d
fo  f o
fo f o o
EOF

./prepare_lang.py --lang-dir $lang_dir --debug 1
./generate_unique_lexicon.py --lang-dir $lang_dir

cat <<EOF > $lang_dir/transcript_words.txt
foo bar bark food food2 fo f a foo bar
bar food2 fo bark
EOF

./convert_transcript_words_to_tokens.py \
  --lexicon $lang_dir/uniq_lexicon.txt \
  --transcript $lang_dir/transcript_words.txt \
  --oov "<UNK>" \
  > $lang_dir/transcript_tokens.txt

../shared/make_kn_lm.py \
  -ngram-order 2 \
  -text $lang_dir/transcript_tokens.txt \
  -lm $lang_dir/P.arpa

echo "Please delete the directory '$lang_dir' manually"
