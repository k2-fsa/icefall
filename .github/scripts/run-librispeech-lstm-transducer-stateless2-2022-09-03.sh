#!/usr/bin/env bash
#
set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

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

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"

if [[ x"${GITHUB_EVENT_LABEL_NAME}" == x"shallow-fusion" ]]; then
  lm_repo_url=https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
  log "Download pre-trained RNN-LM model from ${lm_repo_url}"
  GIT_LFS_SKIP_SMUDGE=1 git clone $lm_repo_url
  lm_repo=$(basename $lm_repo_url)
  pushd $lm_repo
  git lfs pull --include "exp/pretrained.pt"
  mv exp/pretrained.pt exp/epoch-88.pt
  popd

  mkdir -p lstm_transducer_stateless2/exp
  ln -sf $PWD/$repo/exp/pretrained.pt lstm_transducer_stateless2/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh lstm_transducer_stateless2/exp

  log "Decoding test-clean and test-other with RNN LM"

  ./lstm_transducer_stateless2/decode.py \
    --use-averaged-model 0 \
    --epoch 999 \
    --avg 1 \
    --exp-dir lstm_transducer_stateless2/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam 4 \
    --use-shallow-fusion 1 \
    --lm-type rnn \
    --lm-exp-dir $lm_repo/exp \
    --lm-epoch 88 \
    --lm-avg 1 \
    --lm-scale 0.3 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1
fi

if [[ x"${GITHUB_EVENT_LABEL_NAME}" == x"LODR" ]]; then
  bigram_repo_url=https://huggingface.co/marcoyang/librispeech_bigram
  log "Download bi-gram LM from ${bigram_repo_url}"
  GIT_LFS_SKIP_SMUDGE=1 git clone $bigram_repo_url
  bigramlm_repo=$(basename $bigram_repo_url)
  pushd $bigramlm_repo
  git lfs pull --include "2gram.fst.txt"
  cp 2gram.fst.txt $abs_repo/data/lang_bpe_500/.
  popd

  lm_repo_url=https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
  log "Download pre-trained RNN-LM model from ${lm_repo_url}"
  GIT_LFS_SKIP_SMUDGE=1 git clone $lm_repo_url
  lm_repo=$(basename $lm_repo_url)
  pushd $lm_repo
  git lfs pull --include "exp/pretrained.pt"
  mv exp/pretrained.pt exp/epoch-88.pt
  popd

  mkdir -p lstm_transducer_stateless2/exp
  ln -sf $PWD/$repo/exp/pretrained.pt lstm_transducer_stateless2/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh lstm_transducer_stateless2/exp

  log "Decoding test-clean and test-other"

  ./lstm_transducer_stateless2/decode.py \
    --use-averaged-model 0 \
    --epoch 999 \
    --avg 1 \
    --exp-dir lstm_transducer_stateless2/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search_LODR \
    --beam 4 \
    --use-shallow-fusion 1 \
    --lm-type rnn \
    --lm-exp-dir $lm_repo/exp \
    --lm-scale 0.4 \
    --lm-epoch 88 \
    --rnn-lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1 \
    --tokens-ngram 2 \
    --ngram-lm-scale -0.16
fi

if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_NAME}" == x"workflow_dispatch" ]]; then
  mkdir -p lstm_transducer_stateless2/exp
  ln -s $PWD/$repo/exp/pretrained.pt lstm_transducer_stateless2/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh lstm_transducer_stateless2/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in greedy_search fast_beam_search; do
    log "Decoding with $method"

    ./lstm_transducer_stateless2/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --max-duration $max_duration \
      --exp-dir lstm_transducer_stateless2/exp
  done

  rm lstm_transducer_stateless2/exp/*.pt
fi
