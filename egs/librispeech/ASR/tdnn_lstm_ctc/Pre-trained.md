
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
./tdnn_lstm_ctc/pretrained.py --help
```

displays the help information.

## HLG decoding

Once you have the above files ready and have `kaldifeat` installed,
you can run:

```bash
./tdnn_lstm_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  /path/to/your/sound.wav
```

and you will see the transcribed result.

If you want to transcribe multiple files at the same time, you can use:

```bash
./tdnn_lstm_ctc/pretrained.py \
  --checkpoint /path/to/your/checkpoint.pt \
  --words-file /path/to/words.txt \
  --HLG /path/to/HLG.pt \
  /path/to/your/sound1.wav \
  /path/to/your/sound2.wav \
  /path/to/your/sound3.wav
```

**Note**: This is the fastest decoding method.

## HLG decoding + LM rescoring

`./tdnn_lstm_ctc/pretrained.py` also supports `whole lattice LM rescoring`.

To use whole lattice LM rescoring, you also need the following files:

  - G.pt, e.g., `data/lm/G_4_gram.pt` if you have run `./prepare.sh`

The command to run decoding with LM rescoring is:

```bash
./tdnn_lstm_ctc/pretrained.py \
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

# Decoding with a pre-trained model in action

We have uploaded a pre-trained model to <https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc>

The following shows the steps about the usage of the provided pre-trained model.

### (1) Download the pre-trained model

```bash
sudo apt-get install git-lfs
cd /path/to/icefall/egs/librispeech/ASR
git lfs install
mkdir tmp
cd tmp
git clone https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc
```

**CAUTION**: You have to install `git-lfs` to download the pre-trained model.

You will find the following files:

```
tmp/
`-- icefall_asr_librispeech_tdnn-lstm_ctc
    |-- README.md
    |-- data
    |   |-- lang_phone
    |   |   |-- HLG.pt
    |   |   |-- tokens.txt
    |   |   `-- words.txt
    |   `-- lm
    |       `-- G_4_gram.pt
    |-- exp
    |   `-- pretrained.pt
    `-- test_wavs
        |-- 1089-134686-0001.flac
        |-- 1221-135766-0001.flac
        |-- 1221-135766-0002.flac
        `-- trans.txt

6 directories, 10 files
```

**File descriptions**:

  - `data/lang_phone/HLG.pt`

      It is the decoding graph.

  - `data/lang_phone/tokens.txt`

      It contains tokens and their IDs.

  - `data/lang_phone/words.txt`

      It contains words and their IDs.

  - `data/lm/G_4_gram.pt`

      It is a 4-gram LM, useful for LM rescoring.

  - `exp/pretrained.pt`

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from `epoch-14.pt` to `epoch-19.pt`.
      Note: We have removed optimizer `state_dict` to reduce file size.

  - `test_waves/*.flac`

      It contains some test sound files from LibriSpeech `test-clean` dataset.

  - `test_waves/trans.txt`

      It contains the reference transcripts for the sound files in `test_waves/`.

The information of the test sound files is listed below:

```
$ soxi tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/*.flac

Input File     : 'tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac'
Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Duration       : 00:00:06.62 = 106000 samples ~ 496.875 CDDA sectors
File Size      : 116k
Bit Rate       : 140k
Sample Encoding: 16-bit FLAC


Input File     : 'tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac'
Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Duration       : 00:00:16.71 = 267440 samples ~ 1253.62 CDDA sectors
File Size      : 343k
Bit Rate       : 164k
Sample Encoding: 16-bit FLAC


Input File     : 'tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac'
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

./tdnn_lstm_ctc/pretrained.py \
  --checkpoint ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/exp/pretraind.pt \
  --words-file ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/words.txt \
  --HLG ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt \
  ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac \
  ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac \
  ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac
```

The output is given below:

```
2021-08-24 16:57:13,315 INFO [pretrained.py:168] device: cuda:0
2021-08-24 16:57:13,315 INFO [pretrained.py:170] Creating model
2021-08-24 16:57:18,331 INFO [pretrained.py:182] Loading HLG from ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt
2021-08-24 16:57:27,581 INFO [pretrained.py:199] Constructing Fbank computer
2021-08-24 16:57:27,584 INFO [pretrained.py:209] Reading sound files: ['./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac', './tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac', './tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac']
2021-08-24 16:57:27,599 INFO [pretrained.py:215] Decoding started
2021-08-24 16:57:27,791 INFO [pretrained.py:245] Use HLG decoding
2021-08-24 16:57:28,098 INFO [pretrained.py:266]
./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac:
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac:
GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac:
YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


2021-08-24 16:57:28,099 INFO [pretrained.py:268] Decoding Done
```

### (3) Use HLG decoding + LM rescoring

```bash
./conformer_ctc/pretrained.py \
  --checkpoint ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/exp/pretraind.pt \
  --words-file ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/words.txt \
  --HLG ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt \
  --method whole-lattice-rescoring \
  --G ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lm/G_4_gram.pt \
  --ngram-lm-scale 0.8 \
  ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac \
  ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac \
  ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac
```

The output is:

```
2021-08-24 16:39:24,725 INFO [pretrained.py:168] device: cuda:0
2021-08-24 16:39:24,725 INFO [pretrained.py:170] Creating model
2021-08-24 16:39:29,403 INFO [pretrained.py:182] Loading HLG from ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt
2021-08-24 16:39:40,631 INFO [pretrained.py:190] Loading G from ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lm/G_4_gram.pt
2021-08-24 16:39:53,098 INFO [pretrained.py:199] Constructing Fbank computer
2021-08-24 16:39:53,107 INFO [pretrained.py:209] Reading sound files: ['./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac', './tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac', './tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac']
2021-08-24 16:39:53,121 INFO [pretrained.py:215] Decoding started
2021-08-24 16:39:53,443 INFO [pretrained.py:250] Use HLG decoding + LM rescoring
2021-08-24 16:39:54,010 INFO [pretrained.py:266]
./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac:
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac:
GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac:
YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


2021-08-24 16:39:54,010 INFO [pretrained.py:268] Decoding Done
```

**NOTE**: We provide a colab notebook for demonstration.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kNmDXNMwREi0rZGAOIAOJo93REBuOTcd?usp=sharing)

Due to limited memory provided by Colab, you have to upgrade to Colab Pro to run `HLG decoding + LM rescoring`.
Otherwise, you can only run `HLG decoding` with Colab.
