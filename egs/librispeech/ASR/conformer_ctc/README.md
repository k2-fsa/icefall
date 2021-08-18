
# How to use a pre-trained model to transcript a sound file

You need to prepare 4 files:

  - a model checkpoint file, e.g., epoch-20.pt
  - HLG.pt, the decoding graph
  - words.txt, the word symbol table
  - a sound file, whose sampling rate has to be 16 kHz
    Supported formats are those supported by `torchaudio.load()`,
    e.g., wav and flac.

Also, you need to install `kaldifeat`. Please refer to
<https://github.com/csukuangfj/kaldifeat> for installation.

Once you have the above files ready and have `kaldifeat` installed,
you can run:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --hlg /path/to/HLG.pt \
  /path/to/your/sound.wav
```

and you will see the transcribed result.

If you want to transcribe multiple files at the same time, you can use:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --hlg /path/to/HLG.pt \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav \
```
