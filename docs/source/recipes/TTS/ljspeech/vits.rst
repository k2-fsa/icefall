VITS
===============

This tutorial shows you how to train an VITS model
with the `LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_ dataset.

.. note::
  
   TTS related recipes require packages in ``requirements-tts.txt``.

.. note::

   The VITS paper: `Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech <https://arxiv.org/pdf/2106.06103.pdf>`_


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
      --tokens data/tokens.txt
      --max-duration 500

.. note::

    You can adjust the hyper-parameters to control the size of the VITS model and
    the training configurations. For more details, please run ``./vits/train.py --help``.

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

Currently we only support ONNX model exporting. It will generate two files in the given ``exp-dir``:
``vits-epoch-*.onnx`` and ``vits-epoch-*.int8.onnx``.

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

  - `<https://huggingface.co/Zengwei/icefall-tts-ljspeech-vits-2023-11-29>`_
