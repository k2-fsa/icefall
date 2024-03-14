VITS-LJSpeech
===============

This tutorial shows you how to train an VITS model
with the `LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_ dataset.

.. note::

   TTS related recipes require packages in ``requirements-tts.txt``.

.. note::

   The VITS paper: `Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech <https://arxiv.org/pdf/2106.06103.pdf>`_


Install extra dependencies
--------------------------

.. code-block:: bash

  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
  pip install numba espnet_tts_frontend

Data preparation
----------------

.. code-block:: bash

  $ cd egs/ljspeech/TTS
  $ ./prepare.sh

To run stage 1 to stage 5, use

.. code-block:: bash

  $ ./prepare.sh --stage 1 --stop_stage 5


Build Monotonic Alignment Search
--------------------------------

.. code-block:: bash

  $ ./prepare.sh --stage -1 --stop_stage -1

or

.. code-block:: bash

  $ cd vits/monotonic_align
  $ python setup.py build_ext --inplace
  $ cd ../../


Training
--------

.. code-block:: bash

  $ export CUDA_VISIBLE_DEVICES="0,1,2,3"
  $ ./vits/train.py \
      --world-size 4 \
      --num-epochs 1000 \
      --start-epoch 1 \
      --use-fp16 1 \
      --exp-dir vits/exp \
      --tokens data/tokens.txt \
      --model-type high \
      --max-duration 500

.. note::

    You can adjust the hyper-parameters to control the size of the VITS model and
    the training configurations. For more details, please run ``./vits/train.py --help``.

.. warning::

   If you want a model that runs faster on CPU, please use ``--model-type low``
   or ``--model-type medium``.

.. note::

    The training can take a long time (usually a couple of days).

Training logs, checkpoints and tensorboard logs are saved in ``vits/exp``.


Inference
---------

The inference part uses checkpoints saved by the training part, so you have to run the
training part first. It will save the ground-truth and generated wavs to the directory
``vits/exp/infer/epoch-*/wav``, e.g., ``vits/exp/infer/epoch-1000/wav``.

.. code-block:: bash

  $ export CUDA_VISIBLE_DEVICES="0"
  $ ./vits/infer.py \
      --epoch 1000 \
      --exp-dir vits/exp \
      --tokens data/tokens.txt \
      --max-duration 500

.. note::

    For more details, please run ``./vits/infer.py --help``.


Export models
-------------

Currently we only support ONNX model exporting. It will generate one file in the given ``exp-dir``:
``vits-epoch-*.onnx``.

.. code-block:: bash

  $ ./vits/export-onnx.py \
      --epoch 1000 \
      --exp-dir vits/exp \
      --tokens data/tokens.txt

You can test the exported ONNX model with:

.. code-block:: bash

  $ ./vits/test_onnx.py \
      --model-filename vits/exp/vits-epoch-1000.onnx \
      --tokens data/tokens.txt


Download pretrained models
--------------------------

If you don't want to train from scratch, you can download the pretrained models
by visiting the following link:

  - ``--model-type=high``: `<https://huggingface.co/Zengwei/icefall-tts-ljspeech-vits-2024-02-28>`_
  - ``--model-type=medium``: `<https://huggingface.co/csukuangfj/icefall-tts-ljspeech-vits-medium-2024-03-12>`_
  - ``--model-type=low``: `<https://huggingface.co/csukuangfj/icefall-tts-ljspeech-vits-low-2024-03-12>`_

Usage in sherpa-onnx
--------------------

The following describes how to test the exported ONNX model in `sherpa-onnx`_.

.. hint::

   `sherpa-onnx`_ supports different programming languages, e.g., C++, C, Python,
   Kotlin, Java, Swift, Go, C#, etc. It also supports Android and iOS.

   We only describe how to use pre-built binaries from `sherpa-onnx`_ below.
   Please refer to `<https://k2-fsa.github.io/sherpa/onnx/>`_
   for more documentation.

Install sherpa-onnx
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install sherpa-onnx

To check that you have installed `sherpa-onnx`_ successfully, please run:

.. code-block:: bash

   which sherpa-onnx-offline-tts
   sherpa-onnx-offline-tts --help

Download lexicon files
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd /tmp
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
   tar xf espeak-ng-data.tar.bz2

Run sherpa-onnx
^^^^^^^^^^^^^^^

.. code-block:: bash

  cd egs/ljspeech/TTS

  sherpa-onnx-offline-tts \
    --vits-model=vits/exp/vits-epoch-1000.onnx \
    --vits-tokens=data/tokens.txt \
    --vits-data-dir=/tmp/espeak-ng-data \
    --num-threads=1 \
    --output-filename=./high.wav \
    "Ask not what your country can do for you; ask what you can do for your country."

.. hint::

   You can also use ``sherpa-onnx-offline-tts-play`` to play the audio
   as it is generating.

You should get a file ``high.wav`` after running the above command.

Congratulations! You have successfully trained and exported a text-to-speech
model and run it with `sherpa-onnx`_.
