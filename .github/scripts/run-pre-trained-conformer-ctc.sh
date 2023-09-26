#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://github.com/csukuangfj/icefall-asr-conformer-ctc-bpe-500
git lfs install

log "Downloading pre-trained model from $repo_url"
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.flac

log "CTC decoding"

./conformer_ctc/pretrained.py \
  --method ctc-decoding \
  --num-classes 500 \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac

log "HLG decoding"

./conformer_ctc/pretrained.py \
  --method 1best \
  --num-classes 500 \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --words-file $repo/data/lang_bpe_500/words.txt \
  --HLG $repo/data/lang_bpe_500/HLG.pt \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac

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

./local/prepare_lang_fst.py  --lang-dir $repo/data/lang_bpe_500
ls -lh $repo/data/lang_bpe_500

log "Decoding with H on CPU with OpenFst"

./conformer_ctc/jit_pretrained_decode_with_H.py \
  --nn-model $repo/exp/cpu_jit.pt \
  --H $repo/data/lang_bpe_500/H.fst \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac

log "Decoding with HL on CPU with OpenFst"

./conformer_ctc/jit_pretrained_decode_with_HL.py \
  --nn-model $repo/exp/cpu_jit.pt \
  --HL $repo/data/lang_bpe_500/HL.fst \
  --words $repo/data/lang_bpe_500/words.txt \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac

log "Decoding with HLG on CPU with OpenFst"

./conformer_ctc/jit_pretrained_decode_with_HLG.py \
  --nn-model $repo/exp/cpu_jit.pt \
  --HLG $repo/data/lang_bpe_500/HLG.fst \
  --words $repo/data/lang_bpe_500/words.txt \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac
