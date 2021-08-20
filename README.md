
# Table of Contents

- [Installation](#installation)
  * [Install k2](#install-k2)
  * [Install lhotse](#install-lhotse)
  * [Install icefall](#install-icefall)
- [Run recipes](#run-recipes)

## Installation

`icefall` depends on [k2][k2] for FSA operations and [lhotse][lhotse] for
data preparations. To use `icefall`, you have to install its dependencies first.
The following subsections describe how to setup the environment.

CAUTION: There are various ways to setup the environment. What we describe
here is just one alternative.

### Install k2

Please refer to [k2's installation documentation][k2-install] to install k2.
If you have any issues about installing k2, please open an issue at
<https://github.com/k2-fsa/k2/issues>.

### Install lhotse

Please refer to [lhotse's installation documentation][lhotse-install] to install
lhotse.

### Install icefall

`icefall` is a set of Python scripts. What you need to do is just to set
the environment variable `PYTHONPATH`:

```bash
cd $HOME/open-source
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=$HOME/open-source/icefall:$PYTHONPATHON
```

To verify `icefall` was installed successfully, you can run:

```bash
python3 -c "import icefall; print(icefall.__file__)"
```

It should print the path to `icefall`.

## Run recipes

At present, only LibriSpeech recipe is provided. Please
follow [egs/librispeech/ASR/README.md][LibriSpeech] to run it.

## Use Pre-trained models

See [egs/librispeech/ASR/conformer_ctc/README.md](egs/librispeech/ASR/conformer_ctc/README.md)
for how to use pre-trained models.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1huyupXAcHsUrKaWfI83iMEJ6J0Nh0213?usp=sharing)


[LibriSpeech]: egs/librispeech/ASR/README.md
[k2-install]: https://k2.readthedocs.io/en/latest/installation/index.html#
[k2]: https://github.com/k2-fsa/k2
[lhotse]: https://github.com/lhotse-speech/lhotse
[lhotse-install]: https://lhotse.readthedocs.io/en/latest/getting-started.html#installation
