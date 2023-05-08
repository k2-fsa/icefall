## Results

### TIMIT training results (Tdnn_LSTM_CTC)
#### 2021-11-16
(Mingshuang Luo): Result of https://github.com/k2-fsa/icefall/pull/114

TensorBoard log is available at https://tensorboard.dev/experiment/qhA1o025Q322kO34SlhWzg/#scalars

Pretrained model is available at https://huggingface.co/luomingshuang/icefall_asr_timit_tdnn_lstm_ctc

The best decoding results (PER) are listed below, we got this results by averaging models from epoch 16 to 25, and using `whole-lattice-rescoring` with lm_scale equals to 0.08.

||TEST|
|--|--|
|PER| 19.71% |

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

cd egs/timit/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0"
python tdnn_lstm_ctc/train.py --bucketing-sampler True \
                              --concatenate-cuts False \
                              --max-duration 200 \
                              --world-size 1 \
                              --lang-dir data/lang_phone

python tdnn_lstm_ctc/decode.py --epoch 25 \
                               --avg 10 \
                               --max-duration 20 \
                               --lang-dir data/lang_phone
```

### TIMIT training results (Tdnn_LiGRU_CTC)
#### 2021-11-16

(Mingshuang Luo): Result of phone based Tdnn_LiGRU_CTC model, https://github.com/k2-fsa/icefall/pull/114

TensorBoard log is available at https://tensorboard.dev/experiment/IlQxeq5vQJ2SEVP94Y5fyg/#scalars

Pretrained model is available at https://huggingface.co/luomingshuang/icefall_asr_timit_tdnn_ligru_ctc

The best decoding results (PER) are listed below, we got this results by averaging models from epoch 9 to 25, and using `whole-lattice-rescoring` decoding method with lm_scale equals to 0.1.

||TEST|
|--|--|
|PER| 17.66% |

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

cd egs/timit/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0"
python tdnn_ligru_ctc/train.py --bucketing-sampler True \
                              --concatenate-cuts False \
                              --max-duration 200 \
                              --world-size 1 \
                              --lang-dir data/lang_phone

python tdnn_ligru_ctc/decode.py --epoch 25 \
                               --avg 17 \
                               --max-duration 20 \
                               --lang-dir data/lang_phone
```
