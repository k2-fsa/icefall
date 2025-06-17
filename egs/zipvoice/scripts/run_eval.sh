#!/usr/bin/env bash

export PYTHONPATH=../../../:$PYTHONPATH

stage=1
stop_stage=10
generated_wav_path="flow-matching/exp/generated_wavs"

. shared/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le -2 ] && [ $stop_stage -ge -2 ]; then
  log "Stage -2: Install dependencies and download models"

  pip install -r requirements-eval.txt
  pip install git+https://github.com/sarulab-speech/UTMOSv2.git
  modelscope download --model k2-fsa/TTS_eval_models --local_dir TTS_eval_models
fi


if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Prepare evaluation data."

  mkdir -p data/reference/librispeech-test-clean

  gunzip -c data/fbank/librispeech_cuts_with_prompts_test-clean.jsonl.gz | \
    jq -r '"\(.recording.sources[0].source)"' | \
    awk '{split($1, a, "/"); cmd="cp "$1" data/reference/librispeech-test-clean/"a[length(a)]; print cmd; system(cmd)}'


  mkdir -p data/reference/librispeech-test-clean-prompt
  gunzip -c data/fbank/librispeech_cuts_with_prompts_test-clean.jsonl.gz | \
    jq -r '"\(.custom.prompt.recording.sources[0].source) \(.recording.sources[0].source)"' | \
    awk '{split($2, a, "/"); cmd="cp "$1" data/reference/librispeech-test-clean-prompt/"a[length(a)]; print cmd; system(cmd)}'
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Evaluate the model with FSD."

  python local/evaluate_fsd.py --real-path data/reference/librispeech-test-clean \
    --eval-path $generated_wav_path
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Evaluate the model with SIM."

  python local/evaluate_sim.py --real-path data/reference/librispeech-test-clean \
    --eval-path $generated_wav_path
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Evaluate the model with UTMOS."

  python local/evaluate_utmos.py --wav-path $generated_wav_path
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Evaluate the model with UTMOSv2."

  python local/evaluate_utmosv2.py --wav-path $generated_wav_path
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Evaluate the model with WER."

  python local/evaluate_wer_hubert.py --wav-path $generated_wav_path \
    --decode-path $generated_wav_path/decode 
fi
