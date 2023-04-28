#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100
start=0
stop=-1
num_per_split=2000

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  500
)

mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

manifest_dir=data/manifests

mkdir -p $manifest_dir

subset="medium"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Split libri-heavy medium"

  split_dir=data/fbank/libriheavy_medium_split
  mkdir -p $split_dir
  if [ ! -e $split_dir/.split_completed ]; then
    lhotse split-lazy $manifest_dir/librilight_cuts_medium_raw.jsonl.gz $split_dir $num_per_split
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Compute fbank for Libri-heavy small"
  mkdir -p data/fbank
  for part in $subset; do
    log "Processing subset: small"
    if [ ! -e data/fbank/.libriheavy.small.done ]; then
      ./local/compute_fbank_libriheavy.py --dataset $part
      touch data/fbank/.libriheavy.small.done
    fi
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for Libri-heavy medium"
  mkdir -p data/fbank
  num_splits=$(find data/fbank/libriheavy_medium_split -name "librilight_cuts_medium_raw.*.jsonl.gz" | wc -l)
  if [ ! -e data/fbank/.libriheavy.medium.done ]; then
    ./local/compute_fbank_libriheavy.py \
      --dataset medium \
      --num-splits $num_splits \
      --num-workers $nj \
      --start $start \
      --stop $stop
    #touch data/fbank/.libriheavy.medium.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Combine features for medium"
  if [ ! -f data/fbank/librilight_cuts_medium.jsonl.gz ]; then
    pieces=$(find data/fbank/libriheavy_medium_split -name "librilight_cuts_medium.*.jsonl.gz")
    lhotse combine $pieces data/fbank/librilight_cuts_medium.jsonl.gz
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare BPE model"

  tmp_dir=data/tmp
  mkdir -p $tmp_dir
  if [ ! -f $tmp_dir/transcript_words.txt ]; then
    gunzip -c $manifest_dir/librilight_cuts_medium_raw.jsonl.gz |
      jq '.supervisions[].custom.texts[]' | sed 's/" //' | sed 's/\(.*\)"/\1/' > $tmp_dir/transcript_words.txt
  fi

  if [ ! -f $tmp_dir/words.txt ]; then
    cat $tmp_dir/transcript_words.txt | sed 's/ /\n/g' \
      | sort -u | sed '/^$/d' > $tmp_dir/words.txt
    (echo '!SIL'; echo '<SPOKEN_NOISE>'; echo '<UNK>'; ) |
    cat - $tmp_dir/words.txt | sort | uniq | awk '
    BEGIN {
      print "<eps> 0";
    }
    {
      if ($1 == "<s>") {

        print "<s> is in the vocabulary!" | "cat 1>&2"
        exit 1;
      }
      if ($1 == "</s>") {
        print "</s> is in the vocabulary!" | "cat 1>&2"
        exit 1;
      }
      printf("%s %d\n", $1, NR);
    }
    END {
      printf("#0 %d\n", NR+1);
      printf("<s> %d\n", NR+2);
      printf("</s> %d\n", NR+3);
    }' > $tmp_dir/words || exit 1;
    mv $tmp_dir/words $tmp_dir/words.txt
  fi

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    cp $tmp_dir/words.txt $lang_dir/words.txt

    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $tmp_dir/transcript_words.txt
    fi

    done
fi
