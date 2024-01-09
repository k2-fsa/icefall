#!/usr/bin/env bash

conda activate slu_icefall

cd /home/xli257/slu/icefall_st/egs/slu/

# CUDA_VISIBLE_DEVICES=$(free-gpu) python /home/xli257/slu/icefall_st/egs/slu/transducer/pgd_attack.py
# CUDA_VISIBLE_DEVICES=$(free-gpu) python /home/xli257/slu/icefall_st/egs/slu/transducer/pgd_attack_untargeted.py
CUDA_VISIBLE_DEVICES=$(free-gpu) python /home/xli257/slu/icefall_st/egs/slu/transducer/pgd_rank.py
