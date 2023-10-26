#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

dl_dir=$PWD/download

lang_dir=data/lang_phone
lm_dir=data/lm

. shared/parse_options.sh || exit 1

mkdir -p $lang_dir
mkdir -p $lm_dir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"
  mkdir -p $dl_dir

  if [ ! -f $dl_dir/waves_yesno/.completed ]; then
    lhotse download yesno $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare yesno manifest"
  mkdir -p data/manifests
  lhotse prepare yesno $dl_dir/waves_yesno data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for yesno"
  mkdir -p data/fbank
  ./local/compute_fbank_yesno.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare lang"
  # NOTE: "<UNK> SIL" is added for implementation convenience
  # as the graph compiler code requires that there is a OOV word
  # in the lexicon.
  (
    echo "<SIL> SIL"
    echo "YES Y"
    echo "NO N"
    echo "<UNK> SIL"
  ) > $lang_dir/lexicon.txt

  ./local/prepare_lang.py
  ./local/prepare_lang_fst.py --lang-dir ./data/lang_phone --has-silence 1
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare G"
  # We use a unigram G
  cat <<EOF > $lm_dir/G.arpa

\data\\
ngram 1=4

\1-grams:
-1 NO
-1 YES
-99 <s>
-1 </s>

\end\\

EOF

  if [ ! -f $lm_dir/G.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      $lm_dir/G.arpa > $lm_dir/G.fst.txt
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compile HLG"
  if [ ! -f $lang_dir/HLG.pt ]; then
    ./local/compile_hlg.py --lang-dir $lang_dir
  fi
fi
