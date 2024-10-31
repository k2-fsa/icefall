#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/gigaspeech/ASR

repo_url=https://huggingface.co/yfyeung/icefall-asr-gigaspeech-zipformer-2023-10-17

log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "exp/jit_script.pt"
git lfs pull --include "exp/pretrained.pt"
rm epoch-30.pt
ln -s pretrained.pt epoch-30.pt
rm *.onnx
ls -lh
popd

log "----------------------------------------"
log "Export ONNX transducer models "
log "----------------------------------------"

./zipformer/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 30 \
  --avg 1 \
  --exp-dir $repo/exp

ls -lh $repo/exp

log "------------------------------------------------------------"
log "Test exported ONNX transducer models (Python code)          "
log "------------------------------------------------------------"

log "test fp32"
./zipformer/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-30-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-30-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-30-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "test int8"
./zipformer/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-30-avg-1.int8.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-30-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-30-avg-1.int8.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Upload models to huggingface"
git config --global user.name "k2-fsa"
git config --global user.email "xxx@gmail.com"

url=https://huggingface.co/k2-fsa/sherpa-onnx-zipformer-gigaspeech-2023-12-12
GIT_LFS_SKIP_SMUDGE=1 git clone $url
dst=$(basename $url)
cp -v $repo/exp/*.onnx $dst
cp -v $repo/data/lang_bpe_500/tokens.txt $dst
cp -v $repo/data/lang_bpe_500/bpe.model $dst
mkdir -p $dst/test_wavs
cp -v $repo/test_wavs/*.wav $dst/test_wavs
cd $dst
git lfs track "*.onnx"
git add .
git commit -m "upload model" && git push https://k2-fsa:${HF_TOKEN}@huggingface.co/k2-fsa/$dst main || true

log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
rm -rf .git
rm -fv .gitattributes
cd ..
tar cjfv $dst.tar.bz2 $dst
ls -lh
mv -v $dst.tar.bz2 ../../../

log "Export to torchscript model"
./zipformer/export.py \
  --exp-dir $repo/exp \
  --use-averaged-model false \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 30 \
  --avg 1 \
  --jit 1

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.script()"

./zipformer/jit_pretrained.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --nn-model-filename $repo/exp/jit_script.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

for method in greedy_search modified_beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_NAME}" == x"workflow_dispatch" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p zipformer/exp
  ln -s $PWD/$repo/exp/pretrained.pt zipformer/exp/epoch-30.pt
  mkdir -p data
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh zipformer/exp

  mkdir -p data/fbank
  pushd data/fbank

  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/cuts_DEV.jsonl.gz
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/cuts_TEST.jsonl.gz
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/feats_DEV.lca
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/feats_TEST.lca

  ln -sf cuts_DEV.jsonl.gz gigaspeech_cuts_DEV.jsonl.gz
  ln -sf cuts_TEST.jsonl.gz gigaspeech_cuts_TEST.jsonl.gz

  popd

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in greedy_search; do
    log "Decoding with $method"

    ./zipformer/decode.py \
      --decoding-method $method \
      --epoch 30 \
      --avg 1 \
      --use-averaged-model 0 \
      --max-duration $max_duration \
      --exp-dir zipformer/exp
  done

  rm zipformer/exp/*.pt
fi
