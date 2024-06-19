## Results

### whisper_llm_zh finetuning results

|Model|         Training Dataset  | Speech Encoder | LLM |  Projector |
|-| -------------------------| ----------------|------|---------------|
|[yuekai/icefall_asr_aishell_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_aishell_whisper_qwen2_1.5B)  | Aishell1                | whisper-large-v2-aishell1-ft, freeze| Qwen2-1.5B-Instruct, LoRA | Linear, 8x downsample|
| [yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B)  |Multi-hans-zh                | whisper-large-v2-multi-hans-ft, freeze| Qwen2-1.5B-Instruct, LoRA | Linear, 8x downsample||

CER Details:
| Model | [yuekai/icefall_asr_aishell_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_aishell_whisper_qwen2_1.5B) | [yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B) |
|-------|------------------------------------------------|----------------------------------------------------|
| Split | Greedy Search | Greedy Search |
| aishell-1 dev | - | 0.66 |
| aishell-1 test | 3.62 | 0.68 |
| aishell-2 dev | - | 2.67 |
| aishell-2 test | - | 2.94 |
| aishell-4 test | - | 16.20 |
| alimeeting eval | - | 30.86 |
| alimeeting test | - | 40.50 |
| magicdata dev | - | 2.50 |
| magicdata test | - | 1.70 |
| kespeech-asr dev phase1 | - | 6.22 |
| kespeech-asr dev phase2 | - | 2.18 |
| kespeech-asr test | - | 6.59 |
| WenetSpeech dev | - | 4.59 |
| WenetSpeech test_meeting | - | 6.41 |
| WenetSpeech tes_net | - | 6.63 |
| SPEECHIO Avg 001-026 | - | 4.80 |
Command for training is:
```bash
pip install -r whisper_llm_zh/requirements.txt

pip install huggingface_hub['cli']
mkdir -p models/whisper models/qwen

# For aishell fine-tuned whisper model
huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_aishell_whisper exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt
# For multi-hans fine-tuned whisper model
# huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_multi-hans-zh_whisper v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt

# huggingface-clie download  --local-dir models/qwen     Qwen/Qwen2-7B-Instruct
huggingface-clie download  --local-dir models/qwen     Qwen/Qwen2-1.5B-Instruct

torchrun --nproc_per_node 8 ./whisper_llm_zh/train.py \
  --max-duration 200 \
  --exp-dir ./whisper_llm_zh/exp_test \
  --speech-encoder-path-or-name models/whisper/exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt \
  --llm-path-or-name Qwen/Qwen2-1.5B-Instruct \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper_llm_zh/ds_config_zero1.json \
  --use-flash-attn True \
  --use-lora True --unfreeze-llm True
```

Command for decoding using fine-tuned models:
```bash
mkdir -p models/whisper models/qwen models/checkpoint
huggingface-cli download --local-dir models/checkpoint yuekai/icefall_asr_aishell_whisper_qwen2_1.5B

# For aishell fine-tuned whisper model
huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_aishell_whisper exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt
# For multi-hans fine-tuned whisper model
# huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_multi-hans-zh_whisper v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt

huggingface-clie download  --local-dir models/qwen     Qwen/Qwen2-7B-Instruct

mkdir -p whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B
ln -s models/checkpoint/epoch-10-avg-5.pt whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B/epoch-999.pt

python3 ./whisper_llm_zh/decode.py \
  --max-duration 80 \
  --exp-dir whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B \
  --speech-encoder-path-or-name models/whisper/exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt  \
  --llm-path-or-name models/qwen \
  --epoch 999 --avg 1 \
  --manifest-dir data/fbank \
  --use-flash-attn True \
  --use-lora True --dataset aishell
```
