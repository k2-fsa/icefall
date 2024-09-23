#!/usr/bin/env bash

set -ex

git config --global user.name "k2-fsa"
git config --global user.email "csukuangfj@gmail.com"
git config --global lfs.allowincompletepush true

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "pwd: $PWD"

cd egs/multi_zh-hans/ASR

repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-2023-9-2
log "Downloading pre-trained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)
pushd $repo
cd exp
git lfs pull --include pretrained.pt
ln -s pretrained.pt epoch-99.pt
cd ../data/lang_bpe_2000
ls -lh
git lfs pull --include L.pt L_disambig.pt Linv.pt bpe.model
git lfs pull --include "*.model"
ls -lh
popd

log "--------------------------------------------"
log "Export non-streaming ONNX transducer models "
log "--------------------------------------------"
./zipformer/export-onnx.py \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --causal False

ls -lh $repo/exp

./zipformer/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav \
  $repo/test_wavs/TEST_MEETING_T0000000113.wav \
  $repo/test_wavs/TEST_MEETING_T0000000219.wav \
  $repo/test_wavs/TEST_MEETING_T0000000351.wav

rm -rf $repo

repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05
log "Downloading pre-trained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
cd exp/
git lfs pull --include pretrained.pt
rm -fv epoch-20.pt
rm -fv *.onnx
ln -s pretrained.pt epoch-20.pt
cd ../data/lang_bpe_2000
ls -lh
git lfs pull --include L.pt L_disambig.pt Linv.pt bpe.model
git lfs pull --include "*.model"
ls -lh
popd

log "----------------------------------------"
log "Export streaming ONNX CTC models "
log "----------------------------------------"
./zipformer/export-onnx-streaming-ctc.py \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --causal 1 \
  --avg 1 \
  --epoch 20 \
  --use-averaged-model 0 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --use-ctc 1

ls -lh $repo/exp/

log "------------------------------------------------------------"
log "Test exported streaming ONNX CTC models (greedy search)     "
log "------------------------------------------------------------"

test_wavs=(
DEV_T0000000000.wav
DEV_T0000000001.wav
DEV_T0000000002.wav
TEST_MEETING_T0000000113.wav
TEST_MEETING_T0000000219.wav
TEST_MEETING_T0000000351.wav
)

for w in ${test_wavs[@]}; do
  ./zipformer/onnx_pretrained-streaming-ctc.py \
    --model-filename $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/$w
done

log "Upload onnx CTC models to huggingface"
url=https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13
GIT_LFS_SKIP_SMUDGE=1 git clone $url
dst=$(basename $url)
cp -v $repo/exp/ctc*.onnx $dst
cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
cp -v $repo/data/lang_bpe_2000/bpe.model $dst
mkdir -p $dst/test_wavs
cp -v $repo/test_wavs/*.wav $dst/test_wavs
cd $dst
git lfs track "*.onnx" "bpe.model"
ls -lh
file bpe.model
git status
git add .
git commit -m "upload model" && git push https://k2-fsa:${HF_TOKEN}@huggingface.co/k2-fsa/$dst main || true

log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
rm -rf .git
rm -fv .gitattributes
cd ..
tar cjfv $dst.tar.bz2 $dst
ls -lh *.tar.bz2
mv -v $dst.tar.bz2 ../../../

log "----------------------------------------"
log "Export streaming ONNX transducer models "
log "----------------------------------------"

./zipformer/export-onnx-streaming.py \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --causal 1 \
  --avg 1 \
  --epoch 20 \
  --use-averaged-model 0 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --use-ctc 0

ls -lh $repo/exp

log "------------------------------------------------------------"
log "Test exported streaming ONNX transducer models (Python code)"
log "------------------------------------------------------------"

log "test fp32"
./zipformer/onnx_pretrained-streaming.py \
  --encoder-model-filename $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.onnx \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav

log "test int8"
./zipformer/onnx_pretrained-streaming.py \
  --encoder-model-filename $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav

log "Upload onnx transducer models to huggingface"

url=https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12
GIT_LFS_SKIP_SMUDGE=1 git clone $url
dst=$(basename $url)
cp -v $repo/exp/encoder*.onnx $dst
cp -v $repo/exp/decoder*.onnx $dst
cp -v $repo/exp/joiner*.onnx $dst
cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
cp -v $repo/data/lang_bpe_2000/bpe.model $dst
mkdir -p $dst/test_wavs
cp -v $repo/test_wavs/*.wav $dst/test_wavs
cd $dst
git lfs track "*.onnx" bpe.model
git add .
git commit -m "upload model" && git push https://k2-fsa:${HF_TOKEN}@huggingface.co/k2-fsa/$dst main || true

log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
rm -rf .git
rm -fv .gitattributes
cd ..
tar cjfv $dst.tar.bz2 $dst
ls -lh *.tar.bz2
mv -v $dst.tar.bz2 ../../../
