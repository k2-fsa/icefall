## Results
### GigaSpeech BPE training results (Pruned Stateless Transducer 7)

#### [pruned_transducer_stateless7](./pruned_transducer_stateless7)

See #997  for more details.

Number of model parameters: 70369391, i.e., 70.37 M

The best WER, as of 2023-04-17, for Common Voice English 13.0 (cv-corpus-13.0-2023-03-09/en) is below:

Results are:

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy search     | 9.96  | 12.54 |
| modified beam search | 9.86  | 12.48 |

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

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/j4pJQty6RMOkMJtRySREKw/>


### Commonvoice (fr) BPE training results (Pruned Stateless Transducer 7_streaming)

#### [pruned_transducer_stateless7_streaming](./pruned_transducer_stateless7_streaming)

See #1018  for more details.

Number of model parameters: 70369391, i.e., 70.37 M

The best WER for Common Voice French 12.0 (cv-corpus-12.0-2022-12-07/fr) is below:

Results are:

|    decoding method   | Test  |
|----------------------|-------|
|    greedy search     | 9.95  | 
| modified beam search | 9.57  |
|   fast beam search   | 9.67  |

Note: This best result is trained on the full librispeech and gigaspeech, and then fine-tuned on the full commonvoice.

Detailed experimental results and Pretrained model are available at
<https://huggingface.co/shaojieli/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02>

