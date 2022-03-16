#!/usr/bin/env bash

set -eou pipefail

. ./path.sh
. parse_options.sh || exit 1

# Train Conformer CTC model
utils/queue-freegpu.pl --gpu 1 --mem 10G -l "hostname=c*" -q g.q conformer_ctc/exp/decode.log \
  python conformer_ctc/decode.py --epoch 12 --avg 3 --method ctc-decoding --max-duration 50 --num-paths 20
