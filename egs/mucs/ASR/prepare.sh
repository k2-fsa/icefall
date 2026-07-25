#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=60
stage=-1
stop_stage=9

# We assume dl_dir (download dir) contains the following
# directories and files. download them from https://www.openslr.org/resources/104/
#
#  - $dl_dir/hi-en

dl_dir=$PWD/download
mkdir -p $dl_dir

raw_data_path="/data/Database/MUCS/"
dataset="hi-en" #hin-en or bn-en

datadir="data_"$dataset
raw_kaldi_files_path=$dl_dir/$dataset/


. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
vocab_size=400


mkdir -p $datadir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: prepare data files"

  mkdir -p $dl_dir/$dataset
  for x in train dev test train_all; do
  if [ -d "$dl_dir/$dataset/$x" ]; then rm -Rf $dl_dir/$dataset/$x; fi
  done
  mkdir -p $dl_dir/$dataset/{train,test,dev}



  cp -r $raw_data_path/$dataset/"train"/"transcripts"/* $dl_dir/$dataset/"train"
  cp -r $raw_data_path/$dataset/"test"/"transcripts"/* $dl_dir/$dataset/"test"

  for x in train test
    do
    cp $dl_dir/$dataset/$x/"wav.scp" $dl_dir/$dataset/$x/"wav.scp_old"
    cat $dl_dir/$dataset/$x/"wav.scp" | cut -d' ' -f1 > $dl_dir/$dataset/$x/wav_ids
    cat $dl_dir/$dataset/$x/"wav.scp" | cut -d' ' -f2 | awk -v var="$raw_data_path/$dataset/$x/" '{print var$1}' > $dl_dir/$dataset/$x/wav_ids_with_fullpath
    paste -d' ' $dl_dir/$dataset/$x/wav_ids $dl_dir/$dataset/$x/wav_ids_with_fullpath > $dl_dir/$dataset/$x/"wav.scp"
    rm $dl_dir/$dataset/$x/wav_ids
    rm $dl_dir/$dataset/$x/wav_ids_with_fullpath
    done
  ./local/subset_data_dir.sh --first $dl_dir/$dataset/"train" 1000 $dl_dir/$dataset/"dev"
  total=$(wc -l $dl_dir/$dataset/"train"/"text" | cut -d' ' -f1)
  count=$(expr $total - 1000)

  ./local/subset_data_dir.sh --first $dl_dir/$dataset/"train" $count $dl_dir/$dataset/"train_reduced"
  mv $dl_dir/$dataset/"train" $dl_dir/$dataset/"train_all"
  mv $dl_dir/$dataset/"train_reduced" $dl_dir/$dataset/"train"
  

fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: prepare LM files"
  mkdir -p $raw_kaldi_files_path/lm
  if [ ! -e $raw_kaldi_files_path/lm/.done ]; then
    ./local/prepare_lm_files.py --out-dir=$dl_dir/lm --data-path=$raw_kaldi_files_path --mode="train"
    touch $raw_kaldi_files_path/lm/.done
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare MUCS manifest"
  # We assume that you have downloaded the MUCS corpus
  # to $dl_dir/
  mkdir -p $datadir/manifests
  if [ ! -e $datadir/manifests/.mucs.done ]; then
    # generate lhotse manifests from kaldi style files
    ./local/prepare_manifest.py "$raw_kaldi_files_path" $nj $datadir/manifests

    touch $datadir/manifests/.mucs.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for mucs"
  mkdir -p $datadir/fbank
  if [ ! -e $datadir/fbank/.mucs.done ]; then
    ./local/compute_fbank_mucs.py --manifestpath $datadir/manifests/ --fbankpath $datadir/fbank
    touch $datadir/fbank/.mucs.done
  fi

  # exit

  if [ ! -e $datadir/fbank/.mucs-validated.done ]; then
    log "Validating $datadir/fbank for mucs"
    parts=(
      train
      test
      dev
    )
    for part in ${parts[@]}; do
      python3 ./local/validate_manifest.py \
        $datadir/fbank/mucs_cuts_${part}.jsonl.gz
    done
    touch $datadir/fbank/.mucs-validated.done
  fi
fi



if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  lang_dir=$datadir/lang_phone
  mkdir -p $lang_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/mucs_lexicon.txt |
    sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi

  if [ ! -f $lang_dir/L.fst ]; then
    log "Converting L.pt to L.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L.pt \
      $lang_dir/L.fst
  fi

  if [ ! -f $lang_dir/L_disambig.fst ]; then
    log "Converting L_disambig.pt to L_disambig.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L_disambig.pt \
      $lang_dir/disambig_L.fst
  fi
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare BPE based lang"

    lang_dir=$datadir/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp $datadir/lang_phone/words.txt $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      cp download/lm/mucs_vocab_text.txt $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir

      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi

    if [ ! -f $lang_dir/L.fst ]; then
      log "Converting L.pt to L.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L.pt \
        $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
      log "Converting L_disambig.pt to L_disambig.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L_disambig.pt \
        $lang_dir/L_disambig.fst
    fi
  
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Train LM from training data"

    lang_dir=$datadir/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/lm_3.arpa ]; then
      ./shared/make_kn_lm.py \
        -ngram-order 3 \
        -text $lang_dir/transcript_words.txt \
        -lm $lang_dir/lm_3.arpa
    fi

    if [ ! -f $lang_dir/lm_4.arpa ]; then
      ./shared/make_kn_lm.py \
        -ngram-order 4 \
        -text $lang_dir/transcript_words.txt \
        -lm $lang_dir/lm_4.arpa
    fi

fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p $datadir/lm
  if [ ! -f $datadir/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="$datadir/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $datadir/lang_bpe_${vocab_size}/lm_3.arpa > $datadir/lm/G_3_gram.fst.txt
  fi

  if [ ! -f $datadir/lm/G_4_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="$datadir/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $datadir/lang_bpe_${vocab_size}/lm_4.arpa > $datadir/lm/G_4_gram.fst.txt
  fi

fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"

    lang_dir=$datadir/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir

fi

