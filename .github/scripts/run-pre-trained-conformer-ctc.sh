#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

# repo_url=https://github.com/csukuangfj/icefall-asr-conformer-ctc-bpe-500
repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
log "Downloading pre-trained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)
pushd $repo

git lfs pull --include "exp/pretrained.pt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"
git lfs pull --include "data/lang_bpe_500/L.pt"
git lfs pull --include "data/lang_bpe_500/L_disambig.pt"
git lfs pull --include "data/lang_bpe_500/Linv.pt"
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "data/lang_bpe_500/lexicon.txt"
git lfs pull --include "data/lang_bpe_500/lexicon_disambig.txt"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "data/lang_bpe_500/words.txt"
git lfs pull --include "data/lm/G_3_gram.fst.txt"

popd

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

log "CTC decoding"

./conformer_ctc/pretrained.py \
  --method ctc-decoding \
  --num-classes 500 \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "HLG decoding"

./conformer_ctc/pretrained.py \
  --method 1best \
  --num-classes 500 \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --words-file $repo/data/lang_bpe_500/words.txt \
  --HLG $repo/data/lang_bpe_500/HLG.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "CTC decoding on CPU with kaldi decoders using OpenFst"

log "Exporting model with torchscript"

pushd $repo/exp
ln -s pretrained.pt epoch-99.pt
popd

./conformer_ctc/export.py \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --jit 1

ls -lh $repo/exp


log "Generating H.fst, HL.fst"

./local/prepare_lang_fst.py  --lang-dir $repo/data/lang_bpe_500 --ngram-G $repo/data/lm/G_3_gram.fst.txt

ls -lh $repo/data/lang_bpe_500

log "Decoding with H on CPU with OpenFst"

./conformer_ctc/jit_pretrained_decode_with_H.py \
  --nn-model $repo/exp/cpu_jit.pt \
  --H $repo/data/lang_bpe_500/H.fst \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HL on CPU with OpenFst"

./conformer_ctc/jit_pretrained_decode_with_HL.py \
  --nn-model $repo/exp/cpu_jit.pt \
  --HL $repo/data/lang_bpe_500/HL.fst \
  --words $repo/data/lang_bpe_500/words.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG on CPU with OpenFst"

./conformer_ctc/jit_pretrained_decode_with_HLG.py \
  --nn-model $repo/exp/cpu_jit.pt \
  --HLG $repo/data/lang_bpe_500/HLG.fst \
  --words $repo/data/lang_bpe_500/words.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav
