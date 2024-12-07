# Introduction

This is a pseudo-labeling based semi-supervised ASR recipe for the LibriSpeech dataset. The ASR model is Zipformer Transducer. The labeled data is Labeled data is LibriSpeech train-clean-100. Unlabeled data can be LibriSpeech "train-clean-360 + train-other-500" for conventional semi-supervised learning or TedLium3 training set for unsupervised domain adaptation. 

## Description of the recipe

### Preparation of data

The data required in this recipe is the same with LibriSpeech/TedLium3 ASR recipe. And the tokenizer of LibriSpeech is used to build the model. Therefore, we can reuse the `prepare.sh` scripts in those recipes.

### Supervised training for the seed ASR model

Firstly, we need to perform supervised training on the LibriSpeech train-clean-100 subset to generate the seed model for the following pseudo-labeling based semi-supervsed training.

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train_seed.py \
  --world-size 4 \
  --num-epochs 70 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_seed \
  --max-duration 1000
```

For better performance of the seed model, we average the checkpoints as follows:

```
./zipformer/generate_averaged_model.py \
    --epoch 70 \
    --avg 30 \
    --exp-dir ./zipformer/exp_seed
```

The above command generates the final seed model `./zipformer/exp_seed/epoch-70-avg-30.pt`

### Semi-supervised training for the final ASR model

Then, we peform semi-supervised training with the seed model as the initialization. 

- Conventional semi-supervised learning setting where unlabeled data is "train-clean-360 + train-other-500":

```
./zipformer/train_pl.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_pl_librispeech \
  --max-duration 1000 \
  --seed-model-path "zipformer/exp_seed/epoch-70-avg-30.pt" \
  --unlabeled-dataset "librispeech"
```

- Unsupervised domain adaptation setting where unlabeled data is TedLium3 training set:

```
./zipformer/train_pl.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_pl_tedlium \
  --max-duration 1000 \
  --seed-model-path "zipformer/exp_seed/epoch-70-avg-30.pt" \
  --unlabeled-dataset "tedlium"
```

### Decode

Finally, we decode the ASR model to evaluate the performance.

- Evaluate on the LibriSpeech dataset:

```
./zipformer/decode.py \
    --epoch 20 \
    --avg 10 \
    --exp-dir ./zipformer/exp_pl_librispeech \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
    --dataset "librispeech"
```

- Evaluate on the TedLium3 dataset:

```
./zipformer/decode.py \
    --epoch 20 \
    --avg 10 \
    --exp-dir ./zipformer/exp_pl_tedlium \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
    --dataset "tedlium"
```

## Results

- Conventional semi-supervised learning (LibriSpeech 100h/LibriSpeech 860h)

| Model         | test-clean | test-other | comment             |
|-------------------------|------------|------------|---------------------|
| supervised seed model | 5.45       | 13.7      |  --epoch 70 --avg 30 |
| pseudo-labeling model | 4.33       | 9.61      | --epoch 20 --avg 10  |

- Unsupervised domain adaptation (LibriSpeech 100h/TedLium3)

| Model         | tedlium3 dev | tedlium3 test | comment             |
|-------------------------|------------|------------|---------------------|
| supervised seed model | 18.29      | 18.16      |  --epoch 70 --avg 30 |
| pseudo-labeling model | 14.97       | 14.65      | --epoch 20 --avg 10  |


## Pre-trained models and logs

You can find the pre-trained models, training logs, tensorboard logs, decoding logs and decoding results at <https://huggingface.co/zhu-han/icefall-pl-librispeech-zipformer-medium-2023-08-06>
