#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aidatatang_200zh
#      You can find "corpus" and "transcript" inside it.
#      You can download it at
#       https://openslr.org/62/

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  if [ ! -f $dl_dir/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt ]; then
    lhotse download aidatatang-200zh $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aidatatang_200zh manifest"
  # We assume that you have downloaded the aidatatang_200zh corpus
  # to $dl_dir/aidatatang_200zh
  if [ ! -f data/manifests/aidatatang_200zh/.manifests.done ]; then
    mkdir -p data/manifests/aidatatang_200zh
    lhotse prepare aidatatang-200zh $dl_dir data/manifests/aidatatang_200zh
    touch data/manifests/aidatatang_200zh/.manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Process aidatatang_200zh"
  if [ ! -f data/fbank/aidatatang_200zh/.fbank.done ]; then
    mkdir -p data/fbank/aidatatang_200zh
    lhotse prepare aidatatang-200zh $dl_dir data/manifests/aidatatang_200zh
    touch data/fbank/aidatatang_200zh/.fbank.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank for aidatatang_200zh"
  if [ ! -f data/fbank/.aidatatang_200zh.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_aidatatang_200zh.py
    touch data/fbank/.aidatatang_200zh.done
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare char based lang"
  lang_char_dir=data/lang_char
  mkdir -p $lang_char_dir

  # Prepare text.
  grep "\"text\":" data/manifests/aidatatang_200zh/supervisions_train.json \
    | sed -e 's/["text:\t ]*//g' | sed 's/,//g' \
    | ./local/text2token.py -t "char" > $lang_char_dir/text

  # Prepare words.txt
  grep "\"text\":" data/manifests/aidatatang_200zh/supervisions_train.json \
    | sed -e 's/["text:\t]*//g' | sed 's/,//g' \
    | ./local/text2token.py -t "char" > $lang_char_dir/text_words

  cat $lang_char_dir/text_words | sed 's/ /\n/g' | sort -u | sed '/^$/d' \
    | uniq > $lang_char_dir/words_no_ids.txt

  if [ ! -f $lang_char_dir/words.txt ]; then
    ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt
      --output-file $lang_char_dir/words.txt
  fi

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py
  fi
fi
