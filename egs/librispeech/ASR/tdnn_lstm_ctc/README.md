## (To be filled in)

It will contain:

- How to run
- WERs

```bash
cd $PWD/..

./prepare.sh

./tdnn_lstm_ctc/train.py
```

If you have 4 GPUs and want to use GPU 1 and GPU 3 for DDP training,
you can do the following:

```
export CUDA_VISIBLE_DEVICES="1,3"
./tdnn_lstm_ctc/train.py --world-size=2
```
