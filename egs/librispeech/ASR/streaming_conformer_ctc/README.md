## Train and Decode
Commands of data preparation/train/decode steps are almost the same with
../conformer_ctc experiment except some options.

Please read the code and understand following new added options before running this experiment:

  For data preparation:

    Nothing new.

  For streaming_conformer_ctc/train.py:

    --dynamic-chunk-training
    --short-chunk-proportion

  For streaming_conformer_ctc/streaming_decode.py:

    --chunk-size
    --tailing-num-frames
    --simulate-streaming

## Performence and a trained model.

The latest results with this streaming code is shown in following table:

chunk size | wer on test-clean | wer on test-other
-- | -- | --
full | 3.53 | 8.52
40(1.96s) | 3.78 | 9.38
32(1.28s) | 3.82 | 9.44
24(0.96s) | 3.95 | 9.76
16(0.64s) | 4.06 | 9.98
8(0.32s) | 4.30 | 10.55
4(0.16s) | 5.88 | 12.01


A trained model is also provided.
By run
```
git clone https://huggingface.co/GuoLiyong/streaming_conformer

# You may want to manually check md5sum values of downloaded files
# 8e633bc1de37f5ae57a2694ceee32a93  trained_streaming_conformer.pt
# 4c0aeefe26c784ec64873cc9b95420f1  L.pt
# d1f91d81005fb8ce4d65953a4a984ee7  Linv.pt
# e1c1902feb7b9fc69cd8d26e663c2608  bpe.model
# 8617e67159b0ff9118baa54f04db24cc  tokens.txt
# 72b075ab5e851005cd854e666c82c3bb  words.txt
```

If there is any different md5sum values, please run
```
cd streaming_models
git lfs pull
```
And check md5sum values again.

Finally, following files will be downloaded:
<pre>
streaming_models/
|-- lang_bpe
|   |-- L.pt
|   |-- Linv.pt
|   |-- bpe.model
|   |-- tokens.txt
|   `-- words.txt
`-- trained_streaming_conformer.pt
</pre>


And run commands you will get the same results of previous table:
```
trained_models=/path/to/downloaded/streaming_models/
for chunk_size in 4 8 16 24 36 40 -1; do
    ./streaming_conformer_ctc/streaming_decode.py \
      --chunk-size=${chunk_size} \
      --trained-dir=${trained_models}
done
```
Results of following command is indentical to previous one,
but model consumes features chunk_by_chunk, i.e. a streaming way.
```
trained_models=/path/to/downloaded/streaming_models/
for chunk_size in 4 8 16 24 36 40 -1; do
    ./streaming_conformer_ctc/streaming_decode.py \
      --simulate-streaming=True \
      --chunk-size=${chunk_size} \
      --trained-dir=${trained_models}
done
```
