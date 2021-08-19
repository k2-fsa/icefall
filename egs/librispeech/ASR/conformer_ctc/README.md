
# How to use a pre-trained model to transcribe a sound file or multiple sound files

You need to prepare 4 files:

  - a model checkpoint file, e.g., epoch-20.pt
  - HLG.pt, the decoding graph
  - words.txt, the word symbol table
  - a sound file, whose sampling rate has to be 16 kHz
    Supported formats are those supported by `torchaudio.load()`,
    e.g., wav and flac.

Also, you need to install `kaldifeat`. Please refer to
<https://github.com/csukuangfj/kaldifeat> for installation.

```
./conformer_ctc/pretrained.py --help
```

displays the help information.

## HLG decoding

Once you have the above files ready and have `kaldifeat` installed,
you can run:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  /path/to/your/sound.wav
```

and you will see the transcribed result.

If you want to transcribe multiple files at the same time, you can use:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav \
```

**Note**: This is the fastest decoding method.

## HLG decoding + LM rescoring

`./conformer_ctc/pretrained.py` also supports `whole lattice LM rescoring`
and `attention decoder rescoring`.

To use whole lattice LM rescoring, you also need the following files:

  - G.pt, e.g., `data/lm/G_4_gram.pt` if you have run `./prepare.sh`

The command to run decoding with LM rescoring is:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  --method whole-lattice-rescoring \
  --G data/lm/G_4_gram.pt \
  --ngram-lm-scale 0.8 \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav \
```

## HLG Decoding + LM rescoring + attention decoder rescoring

To use attention decoder for rescoring, you need the following extra information:

  - sos token ID
  - eos token ID

The command to run decoding with attention decoder rescoring is:

```
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  --method attention-decoder \
  --G data/lm/G_4_gram.pt \
  --ngram-lm-scale 1.3 \
  --attention-decoder-scale 1.2 \
  --lattice-score-scale 0.5 \
  --num-paths 100 \
  --sos-id 1 \
  --eos-id 1 \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav \
```
