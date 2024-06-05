
export PYTHONPATH=$PYTHONPATH:/workspace/asr/icefall
#pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
#pip install -r whisper_llm_zh/requirements.txt
#export CUDA_VISIBLE_DEVICES=0,1

whisper_path=/workspace/asr/icefall_asr_multi-hans-zh_whisper/v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt
llm_path=/workspace/asr/Qwen1.5-7B-Chat
torchrun --nproc_per_node 8 ./whisper_llm_zh/train.py \
  --max-duration 100 \
  --exp-dir ./whisper_llm_zh/exp_qwen_7b \
  --speech-encoder-path-or-name $whisper_path \
  --llm-path-or-name $llm_path \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper_llm_zh/ds_config_zero1.json