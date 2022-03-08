#!/usr/bin/env bash

set -eou pipefail

. ./path.sh
. parse_options.sh || exit 1

# Train Conformer CTC model
utils/queue-freegpu.pl --gpu 1 --mem 10G -l "hostname=c2[3-7]*" conformer_ctc/exp/train.log \
  python conformer_ctc/train.py --world-size 1
