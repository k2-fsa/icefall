#!/usr/bin/env bash

set -eou pipefail

alignments_dir=data/alignment
cuts_in_dir=data/fbank
cuts_out_dir=data/fbank_ali

python3 ./local/add_alignment_librispeech.py \
  --alignments-dir $alignments_dir \
  --cuts-in-dir $cuts_in_dir \
  --cuts-out-dir $cuts_out_dir
