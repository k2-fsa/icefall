
# How to use a pre-trained model to transcript a sound file

You need to prepare 4 files:

  - a model checkpoint file, e.g., epoch-20.pt
  - HLG.pt, the decoding graph
  - words.txt, the word symbol table
  - a sound file, whose sampling rate has to be 16 kHz
    Supported formats are those supported by `torchaudio.load()`,
    e.g., wav and flac.


Once you have the above files ready, you can run:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --hlg /path/to/HLG.pt \
  --sound-file /path/to/your/sound.wav
```

and you will see the transcribed result.
