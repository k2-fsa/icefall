## Results

### Commonvoice Cantonese (zh-HK) Char training results (Zipformer)

See #1546 for more details.

Number of model parameters:  72526519, i.e., 72.53 M

The best CER, for CommonVoice 16.1 (cv-corpus-16.1-2023-12-06/zh-HK) is below:

|                      |  Dev  | Test |        Note        |
|----------------------|-------|------|--------------------|
|    greedy_search     | 1.17  | 1.22 | --epoch 24 --avg 5 |
| modified_beam_search | 0.98  | 1.11 | --epoch 24 --avg 5 |
|   fast_beam_search   | 1.08  | 1.27 | --epoch 24 --avg 5 |

When doing the cross-corpus validation on [MDCC](https://arxiv.org/abs/2201.02419) (w/o blank penalty),
the best CER is below:

|                      |  Dev  | Test |        Note        |
|----------------------|-------|------|--------------------|
|    greedy_search     | 42.40 | 42.03| --epoch 24 --avg 5 |
| modified_beam_search | 39.73 | 39.19| --epoch 24 --avg 5 |
|   fast_beam_search   | 42.14 | 41.98| --epoch 24 --avg 5 |

When doing the cross-corpus validation on [MDCC](https://arxiv.org/abs/2201.02419) (with blank penalty set to 2.2),
the best CER is below:

|                      |  Dev  | Test |                  Note                  |
|----------------------|-------|------|----------------------------------------|
|    greedy_search     | 39.19 | 39.09| --epoch 24 --avg 5 --blank-penalty 2.2 |
| modified_beam_search | 37.73 | 37.65| --epoch 24 --avg 5 --blank-penalty 2.2 |
|   fast_beam_search   | 37.73 | 37.74| --epoch 24 --avg 5 --blank-penalty 2.2 |

To reproduce the above result, use the following commands for training:

```bash
export CUDA_VISIBLE_DEVICES="0,1"
./zipformer/train_char.py \
  --world-size 2 \
  --num-epochs 30 \
  --start-epoch 1 \ 
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --cv-manifest-dir data/zh-HK/fbank \
  --language zh-HK \
  --use-validated-set 1 \
  --context-size 1 \
  --max-duration 1000
```

and the following commands for decoding:

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode_char.py \
    --epoch 24 \
    --avg 5 \
    --decoding-method $method \
    --exp-dir zipformer/exp \
    --cv-manifest-dir data/zh-HK/fbank \
    --context-size 1 \
    --language zh-HK 
done
```

Detailed experimental results and pre-trained model are available at:
<https://huggingface.co/zrjin/icefall-asr-commonvoice-zh-HK-zipformer-2024-03-20>


### CommonVoice English (en) BPE training results (Pruned Stateless Transducer 7)

#### [pruned_transducer_stateless7](./pruned_transducer_stateless7)

See #997 for more details.

Number of model parameters: 70369391, i.e., 70.37 M

Note that the result is obtained using GigaSpeech transcript trained BPE model

The best WER, as of 2023-04-17, for Common Voice English 13.0 (cv-corpus-13.0-2023-03-09/en) is below:

Results are:

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy_search     | 9.96  | 12.54 |
| modified_beam_search | 9.86  | 12.48 |

To reproduce the above result, use the following commands for training:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./pruned_transducer_stateless7/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \ 
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7/exp \
  --max-duration 550
```

and the following commands for decoding:

```bash
# greedy search
./pruned_transducer_stateless7/decode.py \
  --epoch 30 \
  --avg 5 \
  --decoding-method greedy_search \
  --exp-dir pruned_transducer_stateless7/exp \
  --bpe-model data/en/lang_bpe_500/bpe.model \
  --max-duration 600

# modified beam search
./pruned_transducer_stateless7/decode.py \
  --epoch 30 \
  --avg 5 \
  --decoding-method modified_beam_search \
  --beam-size 4 \
  --exp-dir pruned_transducer_stateless7/exp \
  --bpe-model data/en/lang_bpe_500/bpe.model \
  --max-duration 600
```

Pretrained model is available at
<https://huggingface.co/yfyeung/icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17>

### Commonvoice (fr) BPE training results (Pruned Stateless Transducer 7_streaming)

#### [pruned_transducer_stateless7_streaming](./pruned_transducer_stateless7_streaming)

See #1018  for more details.

Number of model parameters: 70369391, i.e., 70.37 M

The best WER for Common Voice French 12.0 (cv-corpus-12.0-2022-12-07/fr) is below:

Results are:

|    decoding method   | Test  |
|----------------------|-------|
|    greedy_search     | 9.95  | 
| modified_beam_search | 9.57  |
|   fast_beam_search   | 9.67  |

Note: This best result is trained on the full librispeech and gigaspeech, and then fine-tuned on the full commonvoice.

Detailed experimental results and Pretrained model are available at
<https://huggingface.co/shaojieli/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02>

