## Results

### SpeechIO Test Set Decoding Results

##### Decoding results using pretrained [multi-hans-zh zipformer](https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24), [whipser-large-v2](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L27), [whisper-large-v2-wenetspeech-ft](https://huggingface.co/yuekai/icefall_asr_wenetspeech_whisper).

|                        | zipformer_transducer | zipformer_transducer_blank_penalty_2 | whisper_large_v2 | whisper_large_v2_wenetspeech | whisper_large_v2_wenetspeech_zipformer_fusion |
|------------------------|----------------------|--------------------------------------|------------------|------------------------------|-----------------------------------------------|
| SPEECHIO_ASR_ZH00000   | 10.04                | 8.04                                | 11.4             | 9.88                         | 7.78                                          |
| SPEECHIO_ASR_ZH00001   | 1.67                 | 1.51                                | 2.49             | 1.57                         | 1.38                                          |
| SPEECHIO_ASR_ZH00002   | 5.89                 | 5.27                                | 7.89             | 5.65                         | 4.99                                          |
| SPEECHIO_ASR_ZH00003   | 2.66                 | 2.79                                | 5.94             | 2.27                         | 2.33                                          |
| SPEECHIO_ASR_ZH00004   | 3.6                  | 3.34                                | 4.57             | 3.62                         | 3.26                                          |
| SPEECHIO_ASR_ZH00005   | 7.54                 | 5.81                                | 8.39             | 7.26                         | 5.43                                          |
| SPEECHIO_ASR_ZH00006   | 15.59                | 13.34                               | 19.07            | 13.64                        | 11.96                                         |
| SPEECHIO_ASR_ZH00007   | 15.9                 | 15.05                               | 16.7             | 14.06                        | 13.73                                         |
| SPEECHIO_ASR_ZH00008   | 11.07                | 9.68                                | 14.69            | 10.34                        | 8.87                                          |
| SPEECHIO_ASR_ZH00009   | 7.38                 | 6.23                                | 8.32             | 6.74                         | 5.96                                          |
| SPEECHIO_ASR_ZH00010   | 9.19                 | 7.33                                | 11.2             | 8.85                         | 6.97                                          |
| SPEECHIO_ASR_ZH00011   | 4.16                 | 3.84                                | 54.56            | 4.09                         | 3.72                                          |
| SPEECHIO_ASR_ZH00012   | 7.61                 | 6.58                                | 10.53            | 8.35                         | 6.27                                          |
| SPEECHIO_ASR_ZH00013   | 8.72                 | 7.66                                | 9.32             | 7.26                         | 6.7                                           |
| SPEECHIO_ASR_ZH00014   | 9.69                 | 8.71                                | 9.03             | 7.03                         | 6.59                                          |
| SPEECHIO_ASR_ZH00015   | 11.94                | 11.37                               | 16.58            | 12.02                        | 11.11                                         |
| SPEECHIO_ASR_ZH00016   | 9.79                 | 8.79                                | 14.1             | 10.19                        | 8.15                                          |
| SPEECHIO_ASR_ZH00017   | 8                    | 6.72                                | 9.04             | 8.9                          | 6.44                                          |
| SPEECHIO_ASR_ZH00018   | 5.42                 | 5.02                                | 6.06             | 4.86                         | 4.4                                           |
| SPEECHIO_ASR_ZH00019   | 11.26                | 9.06                                | 14.8             | 9.83                         | 8.22                                          |
| SPEECHIO_ASR_ZH00020   | 4.37                 | 4.23                                | 5.97             | 4.23                         | 4.13                                          |
| SPEECHIO_ASR_ZH00021   | 7.81                 | 6.34                                | 8.53             | 7.08                         | 5.88                                          |
| SPEECHIO_ASR_ZH00022   | 9.11                 | 8.54                                | 9.7              | 8.97                         | 8.02                                          |
| SPEECHIO_ASR_ZH00023   | 9.98                 | 8.98                                | 6.31             | 9.44                         | 8.57                                          |
| SPEECHIO_ASR_ZH00024   | 16.15                | 12.95                               | 20.54            | 15.92                        | 12.28                                         |
| SPEECHIO_ASR_ZH00025   | 10.38                | 9.82                                | 11.4             | 10.26                        | 9.27                                          |
| SPEECHIO_ASR_ZH00026   | 5.69                 | 5.63                                | 9.09             | 5.95                         | 5.51                                          |
| Average WER (001-026)           | 8.48                 | 7.48                                | 12.11            | 8.01                         | 6.93                                          |




Command for decoding using fine-tuned whisper:
```bash
git lfs install
git clone https://huggingface.co/yuekai/icefall_asr_wenetspeech_whisper
ln -s icefall_asr_aishell_whisper/exp_large_v2/epoch-4-avg3.pt whisper/exp_large_v2_wenetspeech/epoch-999.pt

python3 ./whisper/decode.py \
  --exp-dir whisper/exp_large_v2_wenetspeech \
  --model-name large-v2 \
  --epoch 999 --avg 1 \
  --start-index 0 --end-index 26 \
  --remove-whisper-encoder-input-length-restriction True \
  --manifest-dir data/fbank \
  --beam-size 1 --max-duration 50
```
Command for decoding using pretrained zipformer:
```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24
cd icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24
git lfs pull --include "exp/pretrained.pt"
git lfs pull --include "data/lang_bpe_2000/*"
ln -s ../icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24/exp/pretrained.pt zipformer/exp_pretrain/epoch-999.pt
ln -s ../icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24/data/lang_bpe_2000/ ./data
wget https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615/resolve/main/data/lang_char/words.txt
mv words.txt ./data/lang_bpe_2000/

./zipformer/decode.py \
    --epoch 999 \
    --avg 1 \
    --blank-penalty 2.0 \
    --use-averaged-model false \
    --exp-dir ./zipformer/exp_pretrain \
    --max-duration 600 \
    --start-index 0 --end-index 26 \
    --manifest-dir data/fbank_kaldi \
    --decoding-method greedy_search
```
Command for fusion the above decoding results from whisper and zipformer:
```bash
python local/whisper_zipformer_fusion.py \
  --whisper-log-dir ./whisper/exp_large_v2_wenetspeech \
  --zipformer-log-dir ./zipformer/exp_pretrain/greedy_search \
  --output-log-dir ./results_fusion

```

See why the fusion helps [here](./local/whisper_zipformer_fusion.py).

SpeechIO fbank features, decoding scripts, logs, and decoding results
are available at
<https://huggingface.co/yuekai/icefall_asr_speechio>
