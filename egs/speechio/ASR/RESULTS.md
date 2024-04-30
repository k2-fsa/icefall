## Results

### SpeechIO Test Set Decoding Results




#### **Unlocked** SpeechIO test sets (ZH00001 ~ ZH00026)
| Rank 排名 | Model 模型 | CER 字错误率 | Date 时间 |
| --- | --- | --- | --- |
| 1 | ximalaya_api_zh | 1.72% | 2023.12 |
| 2 | aliyun_ftasr_api_zh | 1.85% | 2023.12 |
| 3 | microsoft_batch_zh | 2.40% | 2023.12 |
| 4 | bilibili_api_zh | 2.90% | 2023.09 |
| 5 | tencent_api_zh | 3.18% | 2023.12 |
| 6 | iflytek_lfasr_api_zh | 3.32% | 2023.12 |
| 7 | aispeech_api_zh | 3.62% | 2023.12 |
| 8 | **whisper-large-ft-v1** | **4.32%** | 2024.04 |
| 9 | **whisper-large-ft-v0.5** | **4.60%** | 2024.04 |
| 10 | **zipformer (70Mb)** | **6.17%** | 2023.10 |
| 11 | **whisper-large-ft-v0**  | **6.34%** | 2023.03 |
| 12 | baidu_pro_api_zh | 7.29% | 2023.12 |

Note: Above API results are from [SPEECHIO](https://github.com/SpeechColab/Leaderboard). All results used the default [normalize method.](https://github.com/SpeechColab/Leaderboard/blob/master/utils/benchmark.sh#L67)

<details><summary> Detail all models </summary><p>

| Model | Training Set | Note |
|----------------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------|
|[zipformer](https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24)| multi-hans-zh | decoding with transducer head and blank penalty 2.0 |
|[whisper-large-ft-v0](https://huggingface.co/yuekai/icefall_asr_wenetspeech_whisper/tree/main/exp_large_v2)| wenetspeech | greedy_search, 3 epochs|
|[whisper-large-ft-v0.5](https://huggingface.co/yuekai/icefall_asr_wenetspeech_whisper/blob/main/epoch-2-avg-5.pt)| wenetspeech(updated) | [wenetspeech update method](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/fix_manifest.py), greedy_search, 2 epochs |
|[whisper-large-ft-v1](https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper/tree/main/v1.1)|wenetspeech(updated), other multi-hans-zh exclude datatang 200h|[wenetspeech update method](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/fix_manifest.py), greedy search, 3 epochs|

</details>


<details><summary> Detail all results (字错误率 CER %) </summary><p>

| Test Set ID          | 测试场景&内容领域             | bilibili_api_zh (2023.09)  | whisper-large-ft-v0 | whisper-large-ft-v1 | zipformer |
|----------------------|-------------------------------|-----------------|---------|-----------|-----------|
| Avg (01-26)          |                              |  2.9            | 6.34    | 4.32      | 6.17      |
| SPEECHIO_ASR_ZH00001 | 新闻联播                       | 0.54            | 1.42    | 1.09      | 1.37      |
| SPEECHIO_ASR_ZH00002 | 访谈 鲁豫有约                  | 2.78            | 4.76    | 3.21      | 4.67      |
| SPEECHIO_ASR_ZH00003 | 电视节目 天下足球              | 0.81            | 2.17    | 1.70      | 2.71      |
| SPEECHIO_ASR_ZH00004 | 场馆演讲 罗振宇跨年            | 1.48            | 2.53    | 1.86      | 2.54      |
| SPEECHIO_ASR_ZH00005 | 在线教育 李永乐 科普           | 1.47            | 4.27    | 1.95      | 3.12      |
| SPEECHIO_ASR_ZH00006 | 直播 王者荣耀 张大仙&骚白      | 5.85            | 12.55   | 9.46      | 12.86     |
| SPEECHIO_ASR_ZH00007 | 直播 带货 李佳琪&薇娅          | 6.19            | 13.38   | 10.38     | 14.58     |
| SPEECHIO_ASR_ZH00008 | 线下培训 老罗语录              | 3.68            | 9.56    | 6.9      | 9.05      |
| SPEECHIO_ASR_ZH00009 | 播客 故事FM                    | 3.18            | 5.66    | 3.78      | 5.4       |
| SPEECHIO_ASR_ZH00010 | 播客 创业内幕                  | 3.51            | 7.84    | 4.36      | 6.4       |
| SPEECHIO_ASR_ZH00011 | 在线教育 罗翔 刑法法考         | 1.77            | 3.22    | 2.40      | 3.12      |
| SPEECHIO_ASR_ZH00012 | 在线教育 张雪峰 考研           | 2.11            | 5.98    | 3.03      | 4.41      |
| SPEECHIO_ASR_ZH00013 | 短视频 影剪 谷阿莫&牛叔说电影  | 2.97            | 5.91    | 3.72      | 6.56      |
| SPEECHIO_ASR_ZH00014 | 短视频 美式&烹饪               | 3.56            | 6.03    | 4.92      | 8.14      |
| SPEECHIO_ASR_ZH00015 | 评书 单田芳 白眉大侠           | 4.72            | 8.77    | 7.92      | 9.1       |
| SPEECHIO_ASR_ZH00016 | 相声 德云社专场                | 3.01            | 5.24    | 4.15      | 5.59      |
| SPEECHIO_ASR_ZH00017 | 脱口秀 吐槽大会                | 2.93            | 7.05    | 3.04      | 5.17      |
| SPEECHIO_ASR_ZH00018 | 少儿卡通 小猪佩奇&熊出没       | 1.98            | 3.53    | 3.27      | 4.15      |
| SPEECHIO_ASR_ZH00019 | 体育赛事解说 NBA比赛           | 2.32            | 6.89    | 4.39      | 6.66      |
| SPEECHIO_ASR_ZH00020 | 纪录片 篮球人物                | 1.51            | 4.16    | 3.04      | 4.2       |
| SPEECHIO_ASR_ZH00021 | 短视频 汽车之家 汽车评测      | 1.75            | 4.77    | 2.69      | 4.17      |
| SPEECHIO_ASR_ZH00022 | 短视频 小艾大叔 豪宅带看       | 3.29            | 6.35    | 5.44      | 6.72      |
| SPEECHIO_ASR_ZH00023 | 短视频 开箱视频 Zeal&无聊开箱  | 2.18            | 8.99    | 4.08      | 7.94      |
| SPEECHIO_ASR_ZH00024 | 短视频 付老师 农业种植         | 4.80            | 10.81   | 6.06      | 8.64      |
| SPEECHIO_ASR_ZH00025 | 线下课堂 石国鹏 古希腊哲学     | 3.32            | 8.41    | 5.39       | 8.54      |
| SPEECHIO_ASR_ZH00026 | 广播电台节目 张震鬼故事        | 3.70            | 4.52    | 4.06      | 4.67      |
</details>


Command for decoding using fine-tuned whisper:
```bash
git lfs install
git clone https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper
ln -s icefall_asr_multi-hans-zh_whisper/v1.1/epoch-3-avg-10.pt whisper/exp_large_v2/epoch-999.pt

python3 ./whisper/decode.py \
  --exp-dir whisper/exp_large_v2 \
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

SpeechIO fbank features, decoding scripts, logs, and decoding results
are available at [part1](<https://huggingface.co/yuekai/icefall_asr_speechio>) and [part2](https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper/tree/main/v1.1).
