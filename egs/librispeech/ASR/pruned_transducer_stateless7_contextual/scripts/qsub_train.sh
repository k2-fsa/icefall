#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/librispeech/ASR/
#$ -V
#$ -N train_context
#$ -j y -o /exp/rhuang/meta/icefall/egs/librispeech/ASR/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=32G,h_rt=600:00:00,gpu=4,hostname=!r7n07*
#$ -q gpu.q@@v100

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
# hostname
# nvidia-smi

# conda activate /home/hltcoe/rhuang/mambaforge/envs/aligner5
export PATH="/home/hltcoe/rhuang/mambaforge/envs/aligner5/bin/":$PATH
module load cuda11.7/toolkit
module load cudnn/8.5.0.96_cuda11.x
module load nccl/2.13.4-1_cuda11.7
module load gcc/7.2.0
module load intel/mkl/64/2019/5.281

which python
nvcc --version
nvidia-smi
date

# k2
K2_ROOT=/exp/rhuang/meta/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/temp.linux-x86_64-cpython-310/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH

# # torchaudio recipe
# cd /exp/rhuang/meta/audio
# cd examples/asr/librispeech_conformer_ctc

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"
echo "current path:" `pwd`

# export PYTHONPATH=/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2:$PYTHONPATH

# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_ali/exp/exp_libri  # 11073148
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_ali/exp/exp_libri_100  # 11073150
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_ali/exp/exp_libri_100_ts  # 11073234, 11073240, 11073243
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_ali/exp/exp_libri_ts  # 11073238, 11073255, log-train-2024-01-13-20-11-00 => 11073331 => log-train-2024-01-14-02-16-10
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_ali/exp/exp_libri_ts2  # log-train-2024-01-14-07-22-54-0

# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_100
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri2   # baseline, no biasing
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri    # 11169512
# exp_dir=/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy   # 11169515, 11169916
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_34
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_3ctc  # 11171405, log-train-2024-03-04-02-10-27-2, 
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_3ctc  # 11171405, log-train-2024-03-04-21-22-31-0
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_3ctc_attn  #

# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_2early
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_4early
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_234early
# exp_dir=pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_libri_proxy_3early_no5

exp_dir=pruned_transducer_stateless7_contextual/exp/exp_libri_test

mkdir -p $exp_dir

echo
echo "exp_dir:" $exp_dir
echo

path_to_pretrained_asr_model=/exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/
# path_to_pretrained_asr_model=/scratch4/skhudan1/rhuang25/icefall/egs/librispeech/ASR/download/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11

# From pretrained ASR model
if [ ! -f $exp_dir/epoch-1.pt ]; then
  ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $exp_dir/epoch-1.pt
fi

####################################
# train
####################################

# if false; then
#    echo "True"
# else
#    echo "False"
# fi

if true; then
    # # stage 1:
    max_duration=1600
    # max_duration=400  # libri100
    n_distractors=0
    is_full_context=true

    # stage 2:
    # max_duration=1600
    # # max_duration=400  # libri100
    # n_distractors=100
    # is_full_context=false

    python pruned_transducer_stateless7_contextual/train.py \
      --world-size 4 \
      --use-fp16 true \
      --max-duration $max_duration \
      --exp-dir $exp_dir \
      --bpe-model "data/lang_bpe_500/bpe.model" \
      --prune-range 5 \
      --full-libri true \
      --context-dir "data/fbai-speech/is21_deep_bias/" \
      --keep-ratio 1.0 \
      --start-epoch 2 \
      --num-epochs 30 \
      --is-pretrained-context-encoder false \
      --is-reused-context-encoder false \
      --is-full-context $is_full_context \
      --n-distractors $n_distractors  --start-epoch 14 --num-epochs 40 --master-port 12357 --proxy-prob 0.2 --keep-ratio 0.8 --throwaway-prob 0.7 # --start-batch 24000 # --base-lr 0.08  --master-port 12355 --irrelevance-learning true
fi

--n-distractors $n_distractors --master-port 12357 --proxy-prob 0.4 --early-layers 2 --enable-nn true
--n-distractors $n_distractors --master-port 12357 --proxy-prob 0.4 --early-layers 4 --enable-nn true
--n-distractors $n_distractors --master-port 12357 --proxy-prob 0.4 --early-layers 2,3,4 --enable-nn true

####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`
# wandb sync $exp_dir/tensorboard

# https://github.com/k2-fsa/icefall/issues/1298
# python -c "import wandb; wandb.init(project='icefall-asr-gigaspeech-zipformer-2023-10-20')"
# wandb sync zipformer/exp/tensorboard -p icefall-asr-gigaspeech-zipformer-2023-10-20

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
# ssh -L 16006:127.0.0.1:6006 rhuang@test1.hltcoe.jhu.edu
# tensorboard --logdir $exp_dir/tensorboard --port 6006
# http://localhost:16006 

# no-biasing: /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7/exp_500_norm/tensorboard/
# bi_enc: /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/
# single_enc: /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1_single_enc


# exp_dir=pruned_transducer_stateless7_context_ali/exp
# n_distractors=0
# max_duration=1200
# python /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context_ali/train.py       --world-size 1       --use-fp16 true  --max-duration $max_duration       --exp-dir $exp_dir       --bpe-model "data/lang_bpe_500/bpe.model"       --prune-range 5       --use-fp16 true       --context-dir "data/uniphore_contexts/"       --keep-ratio 1.0       --start-epoch 2       --num-epochs 30       --is-bi-context-encoder false       --is-pretrained-context-encoder false       --is-full-context true       --n-distractors $n_distractors


####### debug: RuntimeError: grad_scale is too small, exiting: 8.470329472543003e-22
# encoder_out=rs["encoder_out"]; contexts_h=rs["contexts_h"]; contexts_mask=rs["contexts_mask"]
# queries=encoder_out; contexts=contexts_h; contexts_mask=contexts_mask; need_weights=True
# md = rs["encoder_biasing_adapter"]
# with torch.cuda.amp.autocast(enabled=True):
#     queries = md.proj_in(queries)
#     print("queries:", torch.any(torch.isnan(queries) | torch.isinf(queries)))
#     attn_output, attn_output_weights = md.multihead_attn(queries,contexts,contexts,key_padding_mask=contexts_mask,need_weights=need_weights,)
#     print(torch.any(torch.isnan(attn_output) | torch.isinf(attn_output)))
#     print(torch.any(torch.isnan(attn_output_weights) | torch.isinf(attn_output_weights)))
#     output = md.proj_out(attn_output)
#     print(torch.any(torch.isnan(output) | torch.isinf(output)))

#     encoder_out=rs["encoder_out"]; contexts_h=rs["contexts_h"]; contexts_mask=rs["contexts_mask"]
#     queries=encoder_out; contexts=contexts_h; contexts_mask=contexts_mask; need_weights=True
#     md = rs["encoder_biasing_adapter"]
#     with torch.cuda.amp.autocast(enabled=False):
#         queries = md.proj_in(queries)
#         print("queries:", torch.any(torch.isnan(queries) | torch.isinf(queries)))
#         attn_output, attn_output_weights = md.multihead_attn(queries,contexts,contexts,key_padding_mask=contexts_mask,need_weights=need_weights,)
#         print(torch.any(torch.isnan(attn_output) | torch.isinf(attn_output)))
#         print(torch.any(torch.isnan(attn_output_weights) | torch.isinf(attn_output_weights)))
#         output = md.proj_out(attn_output)
#         print(torch.any(torch.isnan(output) | torch.isinf(output)))
    
#     encoder_out=rs["encoder_out"]; contexts_h=rs["contexts_h"]; contexts_mask=rs["contexts_mask"]
#     queries=encoder_out; contexts=contexts_h; contexts_mask=contexts_mask; need_weights=True
#     md = rs["encoder_biasing_adapter"]
#     queries = md.proj_in(queries)
#     print("queries:", torch.any(torch.isnan(queries) | torch.isinf(queries)))
#     attn_output, attn_output_weights = md.multihead_attn(queries,contexts,contexts,key_padding_mask=contexts_mask,need_weights=need_weights,)
#     print(torch.any(torch.isnan(attn_output) | torch.isinf(attn_output)))
#     print(torch.any(torch.isnan(attn_output_weights) | torch.isinf(attn_output_weights)))
#     output = md.proj_out(attn_output)
#     print(torch.any(torch.isnan(output) | torch.isinf(output)))



python pruned_transducer_stateless7_context_proxy_all_layers/train.py       --world-size 4       --use-fp16 true       --max-duration $max_duration       --exp-dir $exp_dir       --bpe-model "data/lang_bpe_500/bpe.model"       --prune-range 5       --full-libri true       --context-dir "data/fbai-speech/is21_deep_bias/"       --keep-ratio 1.0       --start-epoch 2       --num-epochs 30       --is-pretrained-context-encoder false       --is-reused-context-encoder false       --is-full-context $is_full_context       --n-distractors $n_distractors  --start-epoch 14 --num-epochs 40 --master-port 12357 --proxy-prob 0.2 --keep-ratio 0.8 --throwaway-prob 0.7 --n-distractors 100 --start-epoch 29 --num-epochs 40 --world-size 4 --max-duration 1800