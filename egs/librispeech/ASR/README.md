
## Data preparation

If you want to use `./prepare.sh` to download everything for you,
you can just run

```
./prepare.sh
```

If you have pre-downloaded the LibriSpeech dataset, please
read `./prepare.sh` and modify it to point to the location
of your dataset so that it won't re-download it. After modification,
please run

```
./prepare.sh
```

The script `./prepare.sh` prepares features, lexicon, LMs, etc.
All generated files are saved in the folder `./data`.

**HINT:** `./prepare.sh` supports options `--stage` and `--stop-stage`.

## TDNN-LSTM CTC training

The folder `tdnn_lstm_ctc` contains scripts for CTC training
with TDNN-LSTM models.

Pre-configured parameters for training and decoding are set in the function
`get_params()` within `tdnn_lstm_ctc/train.py`
and `tdnn_lstm_ctc/decode.py`.

Parameters that can be passed from the command-line can be found by

```
./tdnn_lstm_ctc/train.py --help
./tdnn_lstm_ctc/decode.py --help
```

If you have 4 GPUs on a machine and want to use GPU 0, 2, 3 for
mutli-GPU training, you can run

```
export CUDA_VISIBLE_DEVICES="0,2,3"
./tdnn_lstm_ctc/train.py \
  --master-port 12345 \
  --world-size 3
```

If you want to decode by averaging checkpoints `epoch-8.pt`,
`epoch-9.pt` and `epoch-10.pt`, you can run

```
./tdnn_lstm_ctc/decode.py \
  --epoch 10 \
  --avg 3
```

## Conformer CTC training

The folder `conformer-ctc` contains scripts for CTC training
with conformer models. The steps of running the training and
decoding are similar to `tdnn_lstm_ctc`.
