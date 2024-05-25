#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=1
stop_stage=5

data_dir=path/to/fluent/speech/commands
target_root_dir=data/

lang_dir=${target_root_dir}/lang_phone
lm_dir=${target_root_dir}/lm
manifest_dir=${target_root_dir}/manifests
fbanks_dir=${target_root_dir}/fbanks

. shared/parse_options.sh || exit 1

mkdir -p $lang_dir
mkdir -p $lm_dir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "data_dir: $data_dir"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare slu manifest"
  mkdir -p $manifest_dir
  lhotse prepare slu $data_dir $manifest_dir
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for SLU"
  mkdir -p $fbanks_dir
  python ./local/compute_fbank_slu.py $manifest_dir $fbanks_dir
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare lang"
  # NOTE: "<UNK> SIL" is added for implementation convenience
  # as the graph compiler code requires that there is a OOV word
  # in the lexicon.
  python ./local/generate_lexicon.py $data_dir $lm_dir
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Train LM"
  # We use a unigram G
  ./shared/make_kn_lm.py \
    -ngram-order 1 \
    -text $lm_dir/words_transcript.txt \
    -lm $lm_dir/G_transcript.arpa

  ./shared/make_kn_lm.py \
    -ngram-order 1 \
    -text $lm_dir/words_frames.txt \
    -lm $lm_dir/G_frames.arpa

  python ./local/prepare_lang.py $lm_dir

  if [ ! -f $lm_dir/G_transcript.fst.txt ]; then
    python -m kaldilm \
      --read-symbol-table="$lm_dir/words_transcript.txt" \
      $lm_dir/G_transcript.arpa > $lm_dir/G_transcript.fst.txt
  fi

  if [ ! -f $lm_dir/G_frames.fst.txt ]; then
    python -m kaldilm \
      --read-symbol-table="$lm_dir/words_frames.txt" \
      $lm_dir/G_frames.arpa > $lm_dir/G_frames.fst.txt
  fi

  mkdir -p $lm_dir/frames
  mkdir -p $lm_dir/transcript

  chmod -R +777 .

  for i in G_frames.arpa G_frames.fst.txt L_disambig_frames.pt L_frames.pt lexicon_disambig_frames.txt tokens_frames.txt words_frames.txt;
  do
    j=${i//"_frames"/}
    mv "$lm_dir/$i" $lm_dir/frames/$j
  done

  for i in G_transcript.arpa G_transcript.fst.txt L_disambig_transcript.pt L_transcript.pt lexicon_disambig_transcript.txt tokens_transcript.txt words_transcript.txt;
  do
    j=${i//"_transcript"/}
    mv "$lm_dir/$i" $lm_dir/transcript/$j
  done
fi


if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compile HLG"
  ./local/compile_hlg.py --lang-dir $lm_dir/frames
  ./local/compile_hlg.py --lang-dir $lm_dir/transcript

fi
