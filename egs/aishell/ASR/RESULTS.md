## Results

### AIShell training results (Conformer-CTC)
#### 2021-11-17
(Wei Kang): Result of https://github.com/k2-fsa/icefall/pull/30
(Pinfeng Luo): Result of https://github.com/k2-fsa/icefall/pull/123

Pretrained model is available at https://huggingface.co/pfluo/icefall_aishell_model
The tensorboard log for training is available at  https://tensorboard.dev/experiment/zsw6Hn6EQlG8I7HqEkiQpw

The best decoding results (CER) are listed below, we got this results by averaging models from epoch 30 to 49, and using `attention-decoder` decoder with num_paths equals to 100.

||test|
|--|--|
|CER| 4.38% |

||lm_scale|attention_scale|
|--|--|--|
|test|0.6|1.2|

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

cd egs/aishell/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python3 conformer_ctc/train.py --bucketing-sampler False \
                              --concatenate-cuts False \
                              --max-duration 200 \
                              --world-size 8

python3 conformer_ctc/decode.py --lattice-score-scale 0.5 \
                               --epoch 49 \
                               --avg 20 \
                               --method attention-decoder \
                               --max-duration 50 \
                               --num-paths 100
```
