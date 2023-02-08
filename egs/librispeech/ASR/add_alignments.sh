#!/usr/bin/env bash

set -eou pipefail

# align could be in ("mfa", "torchaudio")
# We recommend "torchaudio"
align="torchaudio"

# It adds alignments to the existing fbank features dir (e.g., data/fbank)
# and save cuts to a new dir (e.g., data/fbank_ali).
cuts_in_dir=data/fbank
cuts_out_dir=data/fbank_ali

if [ $align == "mfa" ]; then
  # It add alignments from https://github.com/CorentinJ/librispeech-alignments,
  # generated using the Montreal Forced Aligner (https://montreal-forced-aligner.readthedocs.io).
  alignments_dir=data/alignment

  python3 ./local/add_alignment_librispeech.py \
    --alignments-dir $alignments_dir \
    --cuts-in-dir $cuts_in_dir \
    --cuts-out-dir $cuts_out_dir
elif [ $align == "torchaudio" ]; then
  # See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/bin/modes/workflows.py for details.
  #
  # It use a pretrained ASR model from torchaudio to generate alignments.
  # It will attach word-level alignment information (start, end, and score) to the
  # supervisions in each cut.
  mkdir -p $cuts_out_dir

  parts=(
    train-clean-100
    train-clean-360
    train-other-500
    test-clean
    test-other
    dev-clean
    dev-other
  )

  echo "The alignments will be saved to $cuts_out_dir"
  for part in ${parts[@]}; do
    echo "Start to align $part"
    lhotse workflows align-with-torchaudio --dont-normalize-text \
      $cuts_in_dir/librispeech_cuts_${part}.jsonl.gz \
      $cuts_out_dir/librispeech_cuts_${part}.jsonl.gz
  done
  echo "Finished"
else
  echo "align is expected to be in ('mfa', 'torchaudio'), but got $align"
  exit 1
fi
