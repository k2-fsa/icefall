#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
#. ../../../tools/activate_python.sh

set -eou pipefail

nj=15
stage=-1
stop_stage=100
subset=$1
# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LJSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
#
#  - $dl_dir/lm
#      This directory contains the following files downloaded from
#       http://www.openslr.org/resources/11
#
#        - 3-gram.pruned.1e-7.arpa.gz
#        - 3-gram.pruned.1e-7.arpa
#        - 4-gram.arpa.gz
#        - 4-gram.arpa
#        - LJSpeech-vocab.txt
#        - LJSpeech-lexicon.txt
#        - LJSpeech-lm-norm.txt.gz
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
#dl_dir=/DB/LibriSpeech_tar/vox
dl_dir=/DB/LibriSpeech_tar/$subset
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  5000
  2000
  1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LJSpeech manifest"
  # We assume that you have downloaded the LJSpeech corpus (ver 1.1)
  # You need to prepare LJSpeech according to data_settings/*_list.txt like below
  # $dl_dir/LJSpeech
  # |-- wavs
  # |   |-- train
  # |   |-- dev
  # |   |-- test
  # |-- texts
  # |-- metadata.csv

  # to $dl_dir/LJSpeech
  if [ ! -e $dl_dir/vox/.vox.done ]; then
    #for dset in "4446"; do
    #  log "Resampling vox/$dset set"
    #  file_list=`ls $dl_dir/vox/$dset/`
    #  for wavfile in $file_list; do
	#	echo $wavfile
    #    sox -v 0.9 $dl_dir/vox/$dset/$wavfile -r 16000 -e signed-integer $dl_dir/vox/$dset/tmp_$wavfile
    #    mv $dl_dir/vox/$dset/tmp_$wavfile $dl_dir/vox/$dset/$wavfile
    #  done
    #  log "Resampling $dset done"
    #done
	for dest in "test-clean" "test-other"; do
		for spk in $dl_dir/$dest/*; do
			echo $spk
			spk_id=${spk#*$dest\/}
    		python local/prepare_vox_text.py $spk $spk_id
		done
	done
    #touch $dl_dir/vox/.vox.done
  fi

  mkdir -p data/manifests
  if [ ! -e data/manifests/.vox.done ]; then
	for dest in "test-clean" "test-other"; do
		for spk in $dl_dir/$dest/*; do
			spk_id=${spk#*$dest\/}
			python local/prepare_vox.py $dl_dir/$dest "$spk_id" $subset
		done
	done
    #touch data/manifests/.vox.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  if [ ! -e data/manifests/.musan.done ]; then
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for Vox"
  mkdir -p data/fbank
  rm -rf data/fbank/"$subset"*
  if [ ! -e data/fbank/.LJSpeech.done ]; then
	  for dest in "test-clean" "test-other"; do
		  for spk in $dl_dir/$dest/*; do
			  spk_id=${spk#*$dest\/}
			  ./local/compute_fbank_vox.py --data-dir $spk --spk-id $spk_id --prefix $subset
		  done
	  done
    #touch data/fbank/.vox.done
  fi

  #if [ ! -e data/fbank/.LJSpeech-validated.done ]; then
  #  log "Validating data/fbank for LJSpeech"
  #  parts=`ls $dl_dir/LJSpeech/wavs/`
  #  for part in ${parts[@]}; do
  #    python3 ./local/validate_manifest.py \
  #      data/fbank/LJSpeech_cuts_${part}.jsonl.gz
  #  done
  #fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.musan.done ]; then
    ./local/compute_fbank_musan.py
    #touch data/fbank/.musan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Generate pseudo label"
  rm -rf $dl_dir/*_texts
  for dest in "test-clean" "test-other"; do
	  for spk in $dl_dir/$dest/*; do
		  spk_id=${spk#*$dest\/}
		  echo $spk_id
		  ./pseudo.sh $spk_id $subset
		  #python local/prepare_vox.py $dl_dir/$dest "$spk_id"
	  done
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare LJSpeech manifest"
  # We assume that you have downloaded the LJSpeech corpus (ver 1.1)
  # You need to prepare LJSpeech according to data_settings/*_list.txt like below
  # $dl_dir/LJSpeech
  # |-- wavs
  # |   |-- train
  # |   |-- dev
  # |   |-- test
  # |-- texts
  # |-- metadata.csv

  mkdir -p data/manifests
  if [ ! -e data/manifests/.vox.done ]; then
	for dest in "test-clean" "test-other"; do
		for spk in $dl_dir/$dest/*; do
			spk_id=${spk#*$dest\/}
			python local/prepare_vox.py $dl_dir/$dest "$spk_id" $subset
		done
	done
    #touch data/manifests/.vox.done
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
	log "Stage 7: Re-Compute fbank for Vox"
	mkdir -p data/fbank
	rm -rf data/fbank/"$subset"*
	if [ ! -e data/fbank/.LJSpeech.done ]; then
		for dest in "test-clean" "test-other"; do
			for spk in $dl_dir/$dest/*; do
				spk_id=${spk#*$dest\/}
				./local/compute_fbank_vox.py --data-dir $spk --spk-id $spk_id --speed true --prefix $subset
			done
		done
    #touch data/fbank/.vox.done
	fi
fi