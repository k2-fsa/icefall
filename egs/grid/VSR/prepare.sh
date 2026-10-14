#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=0
stop_stage=9

dl_dir=$PWD/download/grid-corpus

lang_dir=data/lang_phone
lm_dir=data/lm

avhubert_code_dir="$PWD/av_hubert"
# Pre-trained base AvHubert checkpoint (no fine-tuning)
avhubert_ckpts=download/avhubert-ckpts
feats_dir=data/avhubert

. shared/parse_options.sh || exit 1

mkdir -p $lang_dir
mkdir -p $lm_dir
mkdir -p $avhubert_ckpts

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"
  mkdir -p $dl_dir

  if [ ! -f $dl_dir/waves_grid/.completed ]; then
    lhotse download grid $dl_dir
  fi

  dlib_dir=$PWD/download/dlib
  mkdir -p $dlib_dir 

  if [ ! -f $dlib_dir/shape_predictor_68_face_landmarks.dat ]; then
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    mv shape_predictor_68_face_landmarks.dat $dlib_dir    
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare grid manifest"
  mkdir -p data/manifests
  lhotse prepare grid  $dl_dir data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Split grid manifest (train/test)"
  ./local/split_manifests.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute avhubert for grid"
  if [ ! -f "$avhubert_ckpts"/base_vox_iter5.pt ]; then
    log "Downloading AvHubert checkpoint"
	# https://facebookresearch.github.io/av_hubert: AV-HuBERT Base | LRS3 + VoxCeleb2 (En) | No finetuning
	model=https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt
    wget $model -O $avhubert_ckpts/base_vox_iter5.pt
  fi
  
  mkdir -p data/avhubert
  ./local/compute_avhubert_grid.py --avhubert-code-dir ${avhubert_code_dir} \
	  --avhubert-ckpt ${avhubert_ckpts}/base_vox_iter5.pt --feats-dir ${feats_dir} 
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare train text for grid"
  zcat data/manifests/grid_supervisions_train.jsonl.gz | jq -r .text \
	 | sed 's: sp : :g'  > data/lm/train_sentences.txt
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir
  
  if [ ! -f $PWD/download/lexicon.txt ]; then
    curl -OL  https://raw.githubusercontent.com/hmeutzner/kaldi-avsr/refs/heads/master/chime2/input/lexicon.txt
    cat lexicon.txt | tr '[:upper:]' '[:lower:]' > $PWD/download/lexicon.txt
    rm lexicon.txt
  fi

  if [ ! -f $lang_dir/lexicon.txt ]; then
    (echo '!SIL SIL'; echo '<UNK> SPN'; ) |
      cat - $PWD/download/lexicon.txt |
      sort | uniq > $lang_dir/lexicon.txt
  fi

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
      $lang_dir/L_disambig.fst
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare G"
  # Create FST grammar for the GRID
  grammar_cmd="local/create_chime1_grammar.pl"
  lang_dir=data/lang_phone
  $grammar_cmd data/lang_phone/words.txt | fstcompile --keep_isymbols=false --keep_osymbols=false \
   | fstarcsort --sort_type=ilabel \
    > ${lm_dir}/G.fst || exit 1
  fstprint  ${lm_dir}/G.fst > ${lm_dir}/G.fst.txt
  fstprint --isymbols=$lang_dir/words.txt --osymbols=$lang_dir/words.txt ${lm_dir}/G.fst \
   > ${lm_dir}/G.fst.words.txt
  ./local/prepare_lang_fst.py --lang-dir $lang_dir --ngram-G ${lm_dir}/G.fst.txt
  ./local/compile_hlg_phone.py --lang-dir $lang_dir
fi

# bpe vocab size
vocab_size=58

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare BPE based lang"
  lang_dir=data/lang_bpe_${vocab_size}
  mkdir -p $lang_dir
  ./local/train_bpe_model.py \
    --lang-dir $lang_dir \
    --vocab-size ${vocab_size} \
    --transcript  data/lm/train_sentences.txt
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Prepare BPE based lexicon"

    lang_dir=data/lang_bpe_${vocab_size}
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone/words.txt $lang_dir

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

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Generate and Compile LG & HLG"
  lang_dir=data/lang_bpe_${vocab_size}
  
  ./local/prepare_lang_fst.py  \
        --lang-dir $lang_dir \
        --ngram-G ${lm_dir}/G.fst.txt
  
  ./local/compile_hlg.py --lang-dir $lang_dir  
fi
