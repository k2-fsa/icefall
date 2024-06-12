#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

function prepare_data() {
  # We don't download the LM file since it is so large that it will
  # cause OOM error for CI later.
  mkdir -p download/lm
  pushd download/lm
  wget -q https://huggingface.co/csukuangfj/librispeech-for-ci/resolve/main/librispeech-lm-norm.txt.gz
  wget -q https://huggingface.co/csukuangfj/librispeech-for-ci/resolve/main/librispeech-lexicon.txt
  wget -q https://huggingface.co/csukuangfj/librispeech-for-ci/resolve/main/librispeech-vocab.txt
  ls -lh
  gunzip librispeech-lm-norm.txt.gz

  ls -lh
  popd

  pushd download/
  wget -q https://huggingface.co/csukuangfj/librispeech-for-ci/resolve/main/LibriSpeech.tar.bz2
  tar xf LibriSpeech.tar.bz2
  rm LibriSpeech.tar.bz2

  cd LibriSpeech
  ln -s train-clean-100 train-clean-360
  ln -s train-other-500 train-other-500
  popd

  mkdir -p data/manifests

  lhotse prepare librispeech -j 2 -p dev-clean -p dev-other -p test-clean -p test-other -p train-clean-100 download/LibriSpeech data/manifests
  ls -lh data/manifests

  ./local/compute_fbank_librispeech.py --dataset "dev-clean dev-other test-clean test-other train-clean-100" --perturb-speed False
  ls -lh data/fbank

  ./prepare.sh --stage 5 --stop-stage 6
}

function run_diagnostics() {
  ./zipformer/train.py \
    --world-size 1 \
    --num-epochs 1 \
    --start-epoch 1 \
    --use-fp16 0 \
    --exp-dir zipformer/exp-small \
    --causal 0 \
    --num-encoder-layers 1,1,1,1,1,1 \
    --feedforward-dim 64,96,96,96,96,96 \
    --encoder-dim 32,64,64,64,64,64 \
    --encoder-unmasked-dim 32,32,32,32,32,32 \
    --base-lr 0.04 \
    --full-libri 0 \
    --enable-musan 0 \
    --max-duration 30 \
    --print-diagnostics 1
}

function test_streaming_zipformer_ctc_hlg() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  rm $repo/exp-ctc-rnnt-small/*.onnx
  ls -lh $repo/exp-ctc-rnnt-small

  # export models to onnx
  ./zipformer/export-onnx-streaming-ctc.py \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 30 \
    --avg 3 \
    --exp-dir $repo/exp-ctc-rnnt-small \
    --causal 1 \
    --use-ctc 1 \
    --chunk-size 16 \
    --left-context-frames 128 \
    \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192

  ls -lh $repo/exp-ctc-rnnt-small

  for wav in 0.wav 1.wav 8k.wav; do
    python3 ./zipformer/onnx_pretrained_ctc_HLG_streaming.py \
      --nn-model $repo/exp-ctc-rnnt-small/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
      --words $repo/data/lang_bpe_500/words.txt \
      --HLG $repo/data/lang_bpe_500/HLG.fst \
      $repo/test_wavs/$wav
  done

  rm -rf $repo
}

function test_pruned_transducer_stateless_2022_03_12() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless-2022-03-12

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in fast_beam_search modified_beam_search beam_search; do
    log "$method"

    ./pruned_transducer_stateless/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless2_2022_04_29() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless2-2022-04-29

  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  pushd $repo
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/pretrained-epoch-38-avg-10.pt"
  popd

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-epoch-38-avg-10.pt pretrained.pt
  popd

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless2/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless2/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless3_2022_04_29() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29

  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)
  pushd $repo
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/pretrained-epoch-25-avg-6.pt"
  popd

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-epoch-25-avg-6.pt pretrained.pt
  popd

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless3/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless3/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless5_2022_05_13() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless5-2022-05-13

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-epoch-39-avg-7.pt pretrained.pt
  popd

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless5/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --num-encoder-layers 18 \
      --dim-feedforward 2048 \
      --nhead 8 \
      --encoder-dim 512 \
      --decoder-dim 512 \
      --joiner-dim 512 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless5/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav \
      --num-encoder-layers 18 \
      --dim-feedforward 2048 \
      --nhead 8 \
      --encoder-dim 512 \
      --decoder-dim 512 \
      --joiner-dim 512
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless7_2022_11_11() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./pruned_transducer_stateless7/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless7/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless7/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless7/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless8_2022_11_14() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless8/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  log "Export to torchscript model"
  ./pruned_transducer_stateless8/export.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --use-averaged-model false \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless8/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless8/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless8/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless7_ctc_2022_12_01() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01

  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  git lfs pull --include "data/lang_bpe_500/HLG.pt"
  git lfs pull --include "data/lang_bpe_500/L.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  git lfs pull --include "data/lang_bpe_500/Linv.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "data/lm/G_4_gram.pt"
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./pruned_transducer_stateless7_ctc/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless7_ctc/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  for m in ctc-decoding 1best; do
    ./pruned_transducer_stateless7_ctc/jit_pretrained_ctc.py \
      --model-filename $repo/exp/cpu_jit.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --bpe-model $repo/data/lang_bpe_500/bpe.model \
      --G $repo/data/lm/G_4_gram.pt \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless7_ctc/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless7_ctc/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for m in ctc-decoding 1best; do
    ./pruned_transducer_stateless7_ctc/pretrained_ctc.py \
      --checkpoint $repo/exp/pretrained.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --bpe-model $repo/data/lang_bpe_500/bpe.model \
      --G $repo/data/lm/G_4_gram.pt \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_zipformer_mmi_2022_12_08() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-mmi-2022-12-08

  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  git lfs pull --include "data/lang_bpe_500/3gram.pt"
  git lfs pull --include "data/lang_bpe_500/4gram.pt"
  git lfs pull --include "data/lang_bpe_500/L.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  git lfs pull --include "data/lang_bpe_500/Linv.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./zipformer_mmi/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./zipformer_mmi/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    --lang-dir $repo/data/lang_bpe_500 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  for method in 1best nbest nbest-rescoring-LG nbest-rescoring-3-gram nbest-rescoring-4-gram; do
    log "$method"

    ./zipformer_mmi/pretrained.py \
      --method $method \
      --checkpoint $repo/exp/pretrained.pt \
      --lang-dir $repo/data/lang_bpe_500 \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless7_streaming_2022_12_29() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "exp/pretrained.pt"
  git lfs pull --include "exp/encoder_jit_trace.pt"
  git lfs pull --include "exp/decoder_jit_trace.pt"
  git lfs pull --include "exp/joiner_jit_trace.pt"
  cd exp
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./pruned_transducer_stateless7_streaming/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --decode-chunk-len 32 \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless7_streaming/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    --decode-chunk-len 32 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  log "Export to torchscript model by torch.jit.trace()"
  ./pruned_transducer_stateless7_streaming/jit_trace_export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --decode-chunk-len 32 \
    --epoch 99 \
    --avg 1

  log "Decode with models exported by torch.jit.trace()"

  ./pruned_transducer_stateless7_streaming/jit_trace_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --encoder-model-filename $repo/exp/encoder_jit_trace.pt \
    --decoder-model-filename $repo/exp/decoder_jit_trace.pt \
    --joiner-model-filename $repo/exp/joiner_jit_trace.pt \
    --decode-chunk-len 32 \
    $repo/test_wavs/1089-134686-0001.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless7_streaming/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --decode-chunk-len 32 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless7_streaming/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --decode-chunk-len 32 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  rm -rf $repo
}

function test_pruned_transducer_stateless7_ctc_bs_2023_01_29() {
  repo_url=https://huggingface.co/yfyeung/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29

  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  git lfs pull --include "data/lang_bpe_500/L.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  git lfs pull --include "data/lang_bpe_500/HLG.pt"
  git lfs pull --include "data/lang_bpe_500/Linv.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/cpu_jit.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./pruned_transducer_stateless7_ctc_bs/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless7_ctc_bs/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --nn-model-filename $repo/exp/cpu_jit.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  for m in ctc-decoding 1best; do
    ./pruned_transducer_stateless7_ctc_bs/jit_pretrained_ctc.py \
      --model-filename $repo/exp/cpu_jit.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --bpe-model $repo/data/lang_bpe_500/bpe.model \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless7_ctc_bs/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless7_ctc_bs/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for m in ctc-decoding 1best; do
    ./pruned_transducer_stateless7_ctc_bs/pretrained_ctc.py \
      --checkpoint $repo/exp/pretrained.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --bpe-model $repo/data/lang_bpe_500/bpe.model \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_conformer_ctc3_2022_11_27() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conformer-ctc3-2022-11-27

  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  git lfs pull --include "data/lang_bpe_500/HLG.pt"
  git lfs pull --include "data/lang_bpe_500/L.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  git lfs pull --include "data/lang_bpe_500/Linv.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "data/lm/G_4_gram.pt"
  git lfs pull --include "exp/jit_trace.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Decode with models exported by torch.jit.trace()"

  for m in ctc-decoding 1best; do
    ./conformer_ctc3/jit_pretrained.py \
      --model-filename $repo/exp/jit_trace.pt \
      --words-file $repo/data/lang_bpe_500/words.txt \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --bpe-model $repo/data/lang_bpe_500/bpe.model \
      --G $repo/data/lm/G_4_gram.pt \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  log "Export to torchscript model"

  ./conformer_ctc3/export.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --jit-trace 1 \
    --epoch 99 \
    --avg 1 \
    --use-averaged-model 0

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.trace()"

  for m in ctc-decoding 1best; do
    ./conformer_ctc3/jit_pretrained.py \
      --model-filename $repo/exp/jit_trace.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --bpe-model $repo/data/lang_bpe_500/bpe.model \
      --G $repo/data/lm/G_4_gram.pt \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for m in ctc-decoding 1best; do
    ./conformer_ctc3/pretrained.py \
      --checkpoint $repo/exp/pretrained.pt \
      --words-file $repo/data/lang_bpe_500/words.txt \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --G $repo/data/lm/G_4_gram.pt \
      --method $m \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_lstm_transducer_stateless2_2022_09_03() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)
  abs_repo=$(realpath $repo)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-iter-468000-avg-16.pt pretrained.pt
  ln -s pretrained-iter-468000-avg-16.pt epoch-99.pt
  popd

  log "Test exporting with torch.jit.trace()"

  ./lstm_transducer_stateless2/export.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --use-averaged-model 0 \
    --jit-trace 1

  log "Decode with models exported by torch.jit.trace()"

  ./lstm_transducer_stateless2/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --encoder-model-filename $repo/exp/encoder_jit_trace.pt \
    --decoder-model-filename $repo/exp/decoder_jit_trace.pt \
    --joiner-model-filename $repo/exp/joiner_jit_trace.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./lstm_transducer_stateless2/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./lstm_transducer_stateless2/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_pruned_transducer_stateless3_2022_05_13() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-iter-1224000-avg-14.pt pretrained.pt
  ln -s pretrained-iter-1224000-avg-14.pt epoch-99.pt
  popd


  log "Export to torchscript model"
  ./pruned_transducer_stateless3/export.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ./pruned_transducer_stateless3/export.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit-trace 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.trace()"

  ./pruned_transducer_stateless3/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --encoder-model-filename $repo/exp/encoder_jit_trace.pt \
    --decoder-model-filename $repo/exp/decoder_jit_trace.pt \
    --joiner-model-filename $repo/exp/joiner_jit_trace.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  log "Decode with models exported by torch.jit.script()"

  ./pruned_transducer_stateless3/jit_pretrained.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --encoder-model-filename $repo/exp/encoder_jit_script.pt \
    --decoder-model-filename $repo/exp/decoder_jit_script.pt \
    --joiner-model-filename $repo/exp/joiner_jit_script.pt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav


  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless3/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless3/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  rm -rf $repo
}

function test_streaming_pruned_transducer_stateless2_20220625() {
  repo_url=https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless2_20220625

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-epoch-24-avg-10.pt pretrained.pt
  popd

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./pruned_transducer_stateless2/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --simulate-streaming 1 \
      --causal-convolution 1 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./pruned_transducer_stateless2/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --simulate-streaming 1 \
      --causal-convolution 1 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_streaming_zipformer_2023_05_17() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17

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
  git lfs pull --include "exp/jit_script_chunk_16_left_128.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./zipformer/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --causal 1 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  ./zipformer/jit_pretrained_streaming.py \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --nn-model-filename $repo/exp/jit_script_chunk_16_left_128.pt \
    $repo/test_wavs/1089-134686-0001.wav

  for method in greedy_search modified_beam_search fast_beam_search; do
    log "$method"

    ./zipformer/pretrained.py \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_zipformer_2023_05_18() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15

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
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./zipformer/export.py \
    --exp-dir $repo/exp \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
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
  rm -rf $repo
}

function test_transducer_stateless2_torchaudio_2022_04_19() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless2-torchaudio-2022-04-19

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./transducer_stateless2/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in fast_beam_search modified_beam_search beam_search; do
    log "$method"

    ./transducer_stateless2/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_zipformer_transducer_ctc_2023_06_13() {
  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-transducer-ctc-2023-06-13

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
  git lfs pull --include "data/lang_bpe_500/HLG.pt"
  git lfs pull --include "data/lang_bpe_500/L.pt"
  git lfs pull --include "data/lang_bpe_500/LG.pt"
  git lfs pull --include "data/lang_bpe_500/Linv.pt"
  git lfs pull --include "data/lm/G_4_gram.pt"
  git lfs pull --include "exp/jit_script.pt"
  git lfs pull --include "exp/pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "Export to torchscript model"
  ./zipformer/export.py \
    --exp-dir $repo/exp \
    --use-transducer 1 \
    --use-ctc 1 \
    --use-averaged-model false \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --jit 1

  ls -lh $repo/exp/*.pt

  log "Decode with models exported by torch.jit.script()"

  for method in ctc-decoding 1best; do
    ./zipformer/jit_pretrained_ctc.py \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --model-filename $repo/exp/jit_script.pt \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --G $repo/data/lm/G_4_gram.pt \
      --method $method \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in ctc-decoding 1best; do
    log "$method"

    ./zipformer/pretrained_ctc.py \
      --use-transducer 1 \
      --use-ctc 1 \
      --method $method \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      --HLG $repo/data/lang_bpe_500/HLG.pt \
      --G $repo/data/lm/G_4_gram.pt \
      --words-file $repo/data/lang_bpe_500/words.txt  \
      --sample-rate 16000 \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_100h_transducer_stateless_multi_datasets_bpe_500_2022_02_21() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-100h-transducer-stateless-multi-datasets-bpe-500-2022-02-21

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./transducer_stateless_multi_datasets/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./transducer_stateless_multi_datasets/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_transducer_stateless_multi_datasets_bpe_500_2022_03_01() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-multi-datasets-bpe-500-2022-03-01

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./transducer_stateless_multi_datasets/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in modified_beam_search beam_search fast_beam_search; do
    log "$method"

    ./transducer_stateless_multi_datasets/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_transducer_stateless_bpe_500_2022_02_07() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-bpe-500-2022-02-07

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./transducer_stateless/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done

  for method in fast_beam_search modified_beam_search beam_search; do
    log "$method"

    ./transducer_stateless/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_500/tokens.txt \
      $repo/test_wavs/1089-134686-0001.wav \
      $repo/test_wavs/1221-135766-0001.wav \
      $repo/test_wavs/1221-135766-0002.wav
  done
  rm -rf $repo
}

function test_zipformer_ctc_en_2023_10_02() {
  repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-ctc-en-2023-10-02
  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  log "CTC greedy search"

  ./zipformer/onnx_pretrained_ctc.py \
    --nn-model $repo/model.onnx \
    --tokens $repo/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "CTC H decoding"

  ./zipformer/onnx_pretrained_ctc_H.py \
    --nn-model $repo/model.onnx \
    --tokens $repo/tokens.txt \
    --H $repo/H.fst \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "CTC HL decoding"

  ./zipformer/onnx_pretrained_ctc_HL.py \
    --nn-model $repo/model.onnx \
    --words $repo/words.txt \
    --HL $repo/HL.fst \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "CTC HLG decoding"

  ./zipformer/onnx_pretrained_ctc_HLG.py \
    --nn-model $repo/model.onnx \
    --words $repo/words.txt \
    --HLG $repo/HLG.fst \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  rm -rf $repo
}

function test_conformer_ctc_jit_bpe_500_2021_11_09() {
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

  rm -rf $repo
}

function test_transducer_bpe_500_2021_12_23() {
  repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-bpe-500-2021-12-23

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  log "Beam search decoding"

  ./transducer/pretrained.py \
    --method beam_search \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  rm -rf $repo
}

prepare_data
run_diagnostics
test_streaming_zipformer_ctc_hlg
test_pruned_transducer_stateless_2022_03_12
test_pruned_transducer_stateless2_2022_04_29
test_pruned_transducer_stateless3_2022_04_29
test_pruned_transducer_stateless5_2022_05_13
test_pruned_transducer_stateless7_2022_11_11
test_pruned_transducer_stateless8_2022_11_14
test_pruned_transducer_stateless7_ctc_2022_12_01
test_zipformer_mmi_2022_12_08
test_pruned_transducer_stateless7_streaming_2022_12_29
test_pruned_transducer_stateless7_ctc_bs_2023_01_29
test_conformer_ctc3_2022_11_27
test_lstm_transducer_stateless2_2022_09_03
test_pruned_transducer_stateless3_2022_05_13
test_streaming_pruned_transducer_stateless2_20220625
test_streaming_zipformer_2023_05_17
test_zipformer_2023_05_18
test_transducer_stateless2_torchaudio_2022_04_19
test_zipformer_transducer_ctc_2023_06_13
test_100h_transducer_stateless_multi_datasets_bpe_500_2022_02_21
test_transducer_stateless_multi_datasets_bpe_500_2022_03_01
test_transducer_stateless_bpe_500_2022_02_07
test_zipformer_ctc_en_2023_10_02
# test_conformer_ctc_jit_bpe_500_2021_11_09 # failes for torch != 1.13.x and torch != 2.0.x
test_transducer_bpe_500_2021_12_23
