#!/usr/bin/env bash

conda activate slu_icefall

cd /home/xli257/slu/icefall_st/egs/slu/

./transducer/train.py --exp-dir transducer/exp_fscd_align --lang-dir data/fscd_align/lm/frames