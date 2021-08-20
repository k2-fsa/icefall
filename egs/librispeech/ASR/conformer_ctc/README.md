
# How to use a pre-trained model to transcribe a sound file or multiple sound files

(See the bottom of this document for the link to a colab notebook.)

You need to prepare 4 files:

  - a model checkpoint file, e.g., epoch-20.pt
  - HLG.pt, the decoding graph
  - words.txt, the word symbol table
  - a sound file, whose sampling rate has to be 16 kHz.
    Supported formats are those supported by `torchaudio.load()`,
    e.g., wav and flac.

Also, you need to install `kaldifeat`. Please refer to
<https://github.com/csukuangfj/kaldifeat> for installation.

```bash
./conformer_ctc/pretrained.py --help
```

displays the help information.

## HLG decoding

Once you have the above files ready and have `kaldifeat` installed,
you can run:

```bash
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  /path/to/your/sound.wav
```

and you will see the transcribed result.

If you want to transcribe multiple files at the same time, you can use:

```bash
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav
```

**Note**: This is the fastest decoding method.

## HLG decoding + LM rescoring

`./conformer_ctc/pretrained.py` also supports `whole lattice LM rescoring`
and `attention decoder rescoring`.

To use whole lattice LM rescoring, you also need the following files:

  - G.pt, e.g., `data/lm/G_4_gram.pt` if you have run `./prepare.sh`

The command to run decoding with LM rescoring is:

```bash
./conformer_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  --method whole-lattice-rescoring \
  --G data/lm/G_4_gram.pt \
  --ngram-lm-scale 0.8 \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav
```

## HLG Decoding + LM rescoring + attention decoder rescoring

To use attention decoder for rescoring, you need the following extra information:

  - sos token ID
  - eos token ID

The command to run decoding with attention decoder rescoring is:

```bash
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
  /path/to/your/sound3.wav
```

# Decoding with a pre-trained model in action

We have uploaded a pre-trained model to <https://huggingface.co/pkufool/conformer_ctc>

The following shows the steps about the usage of the provided pre-trained model.

### (1) Download the pre-trained model

```bash
sudo apt-get install git-lfs
cd /path/to/icefall/egs/librispeech/ASR
git lfs install
mkdir tmp
cd tmp
git clone https://huggingface.co/pkufool/conformer_ctc
```

**CAUTION**: You have to install `git-lfst` to download the pre-trained model.

You will find the following files:

```
tmp
`-- conformer_ctc
    |-- README.md
    |-- data
    |   |-- lang_bpe
    |   |   |-- HLG.pt
    |   |   |-- bpe.model
    |   |   |-- tokens.txt
    |   |   `-- words.txt
    |   `-- lm
    |       `-- G_4_gram.pt
    |-- exp
    |   `-- pretraind.pt
    `-- test_wavs
        |-- 1089-134686-0001.flac
        |-- 1221-135766-0001.flac
        |-- 1221-135766-0002.flac
        `-- trans.txt

6 directories, 11 files
```

**File descriptions**:

  - `data/lang_bpe/HLG.pt`

      It is the decoding graph.

  - `data/lang_bpe/bpe.model`

      It is a sentencepiece model. You can use it to reproduce our results.

  - `data/lang_bpe/tokens.txt`

      It contains tokens and their IDs, generated from `bpe.model`.
      Provided only for convienice so that you can look up the SOS/EOS ID easily.

  - `data/lang_bpe/words.txt`

      It contains words and their IDs.

  - `data/lm/G_4_gram.pt`

      It is a 4-gram LM, useful for LM rescoring.

  - `exp/pretrained.pt`

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from `epoch-15.pt` to `epoch-34.pt`.
      Note: We have removed optimizer `state_dict` to reduce file size.

  - `test_waves/*.flac`

      It contains some test sound files from LibriSpeech `test-clean` dataset.

  - `test_waves/trans.txt`

      It contains the reference transcripts for the sound files in `test_waves/`.

The information of the test sound files is listed below:

```
$ soxi tmp/conformer_ctc/test_wavs/*.flac

Input File     : 'tmp/conformer_ctc/test_wavs/1089-134686-0001.flac'
Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Duration       : 00:00:06.62 = 106000 samples ~ 496.875 CDDA sectors
File Size      : 116k
Bit Rate       : 140k
Sample Encoding: 16-bit FLAC

Input File     : 'tmp/conformer_ctc/test_wavs/1221-135766-0001.flac'
Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Duration       : 00:00:16.71 = 267440 samples ~ 1253.62 CDDA sectors
File Size      : 343k
Bit Rate       : 164k
Sample Encoding: 16-bit FLAC

Input File     : 'tmp/conformer_ctc/test_wavs/1221-135766-0002.flac'
Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Duration       : 00:00:04.83 = 77200 samples ~ 361.875 CDDA sectors
File Size      : 105k
Bit Rate       : 174k
Sample Encoding: 16-bit FLAC

Total Duration of 3 files: 00:00:28.16
```

### (2) Use HLG decoding

```bash
cd /path/to/icefall/egs/librispeech/ASR

./conformer_ctc/pretrained.py \
  --checkpoint ./tmp/conformer_ctc/exp/pretraind.pt \
  --words-file ./tmp/conformer_ctc/data/lang_bpe/words.txt \
  --HLG ./tmp/conformer_ctc/data/lang_bpe/HLG.pt \
  ./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac \
  ./tmp/conformer_ctc/test_wavs/1221-135766-0001.flac \
  ./tmp/conformer_ctc/test_wavs/1221-135766-0002.flac
```

The output is given below:

```
2021-08-20 11:03:05,712 INFO [pretrained.py:217] device: cuda:0
2021-08-20 11:03:05,712 INFO [pretrained.py:219] Creating model
2021-08-20 11:03:11,345 INFO [pretrained.py:238] Loading HLG from ./tmp/conformer_ctc/data/lang_bpe/HLG.pt
2021-08-20 11:03:18,442 INFO [pretrained.py:255] Constructing Fbank computer
2021-08-20 11:03:18,444 INFO [pretrained.py:265] Reading sound files: ['./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac', './tmp/conformer_ctc/test_wavs/1221-135766-0001.flac', './tmp/conformer_ctc/test_wavs/1221-135766-0002.flac']
2021-08-20 11:03:18,507 INFO [pretrained.py:271] Decoding started
2021-08-20 11:03:18,795 INFO [pretrained.py:300] Use HLG decoding
2021-08-20 11:03:19,149 INFO [pretrained.py:339]
./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac:
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

./tmp/conformer_ctc/test_wavs/1221-135766-0001.flac:
GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED
BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

./tmp/conformer_ctc/test_wavs/1221-135766-0002.flac:
YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


2021-08-20 11:03:19,149 INFO [pretrained.py:341] Decoding Done
```

### (3) Use HLG decoding + LM rescoring

```bash
./conformer_ctc/pretrained.py \
  --checkpoint ./tmp/conformer_ctc/exp/pretraind.pt \
  --words-file ./tmp/conformer_ctc/data/lang_bpe/words.txt \
  --HLG ./tmp/conformer_ctc/data/lang_bpe/HLG.pt \
  --method whole-lattice-rescoring \
  --G ./tmp/conformer_ctc/data/lm/G_4_gram.pt \
  --ngram-lm-scale 0.8 \
  ./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac \
  ./tmp/conformer_ctc/test_wavs/1221-135766-0001.flac \
  ./tmp/conformer_ctc/test_wavs/1221-135766-0002.flac
```

The output is:

```
2021-08-20 11:12:17,565 INFO [pretrained.py:217] device: cuda:0
2021-08-20 11:12:17,565 INFO [pretrained.py:219] Creating model
2021-08-20 11:12:23,728 INFO [pretrained.py:238] Loading HLG from ./tmp/conformer_ctc/data/lang_bpe/HLG.pt
2021-08-20 11:12:30,035 INFO [pretrained.py:246] Loading G from ./tmp/conformer_ctc/data/lm/G_4_gram.pt
2021-08-20 11:13:10,779 INFO [pretrained.py:255] Constructing Fbank computer
2021-08-20 11:13:10,787 INFO [pretrained.py:265] Reading sound files: ['./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac', './tmp/conformer_ctc/test_wavs/1221-135766-0001.flac', './tmp/conformer_ctc/test_wavs/1221-135766-0002.flac']
2021-08-20 11:13:10,798 INFO [pretrained.py:271] Decoding started
2021-08-20 11:13:11,085 INFO [pretrained.py:305] Use HLG decoding + LM rescoring
2021-08-20 11:13:11,736 INFO [pretrained.py:339]
./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac:
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

./tmp/conformer_ctc/test_wavs/1221-135766-0001.flac:
GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED
BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

./tmp/conformer_ctc/test_wavs/1221-135766-0002.flac:
YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


2021-08-20 11:13:11,737 INFO [pretrained.py:341] Decoding Done
```

### (4) Use HLG decoding + LM rescoring + attention decoder rescoring

```bash
./conformer_ctc/pretrained.py \
  --checkpoint ./tmp/conformer_ctc/exp/pretraind.pt \
  --words-file ./tmp/conformer_ctc/data/lang_bpe/words.txt \
  --HLG ./tmp/conformer_ctc/data/lang_bpe/HLG.pt \
  --method attention-decoder \
  --G ./tmp/conformer_ctc/data/lm/G_4_gram.pt \
  --ngram-lm-scale 1.3 \
  --attention-decoder-scale 1.2 \
  --lattice-score-scale 0.5 \
  --num-paths 100 \
  --sos-id 1 \
  --eos-id 1 \
  ./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac \
  ./tmp/conformer_ctc/test_wavs/1221-135766-0001.flac \
  ./tmp/conformer_ctc/test_wavs/1221-135766-0002.flac
```

The output is:

```
2021-08-20 11:19:11,397 INFO [pretrained.py:217] device: cuda:0
2021-08-20 11:19:11,397 INFO [pretrained.py:219] Creating model
2021-08-20 11:19:17,354 INFO [pretrained.py:238] Loading HLG from ./tmp/conformer_ctc/data/lang_bpe/HLG.pt
2021-08-20 11:19:24,615 INFO [pretrained.py:246] Loading G from ./tmp/conformer_ctc/data/lm/G_4_gram.pt
2021-08-20 11:20:04,576 INFO [pretrained.py:255] Constructing Fbank computer
2021-08-20 11:20:04,584 INFO [pretrained.py:265] Reading sound files: ['./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac', './tmp/conformer_ctc/test_wavs/1221-135766-0001.flac', './tmp/conformer_ctc/test_wavs/1221-135766-0002.flac']
2021-08-20 11:20:04,595 INFO [pretrained.py:271] Decoding started
2021-08-20 11:20:04,854 INFO [pretrained.py:313] Use HLG + LM rescoring + attention decoder rescoring
2021-08-20 11:20:05,805 INFO [pretrained.py:339]
./tmp/conformer_ctc/test_wavs/1089-134686-0001.flac:
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

./tmp/conformer_ctc/test_wavs/1221-135766-0001.flac:
GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED
BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

./tmp/conformer_ctc/test_wavs/1221-135766-0002.flac:
YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


2021-08-20 11:20:05,805 INFO [pretrained.py:341] Decoding Done
```

**NOTE**: We provide a colab notebook for demonstration.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1huyupXAcHsUrKaWfI83iMEJ6J0Nh0213?usp=sharing)

Due to limited memory provided by Colab, you have to upgrade to Colab Pro to
run `HLG decoding + LM rescoring` and `HLG decoding + LM rescoring + attention decoder rescoring`.
Otherwise, you can only run `HLG decoding` with Colab.
