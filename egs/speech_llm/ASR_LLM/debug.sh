
export PYTHONPATH=$PYTHONPATH:/mnt/samsung-t7/yuekai/asr/icefall_llm
# pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
# pip install -r whisper/requirements.txt
export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nproc_per_node 2 ./whisper_llm_zh/train.py \
#   --max-duration 80 \
#   --exp-dir ./whisper_llm_zh/exp_test \
#   --speech-encoder-path-or-name tiny \
#   --llm-path-or-name Qwen/Qwen1.5-0.5B-Chat \
#   --manifest-dir data/fbank \
#   --deepspeed \
#   --deepspeed_config ./whisper_llm_zh/ds_config_zero1.json \
#   --use-flash-attn False



python3 ./whisper_llm_zh/decode.py \
  --max-duration 80 \
  --exp-dir ./whisper_llm_zh/exp_test \
  --speech-encoder-path-or-name tiny \
  --llm-path-or-name Qwen/Qwen1.5-0.5B-Chat \
  --epoch 1 --avg 1 \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper_llm_zh/ds_config_zero1.json \
  --use-flash-attn False