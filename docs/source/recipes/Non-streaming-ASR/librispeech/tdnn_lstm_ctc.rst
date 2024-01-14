TDNN-LSTM-CTC
=============

This tutorial shows you how to run a TDNN-LSTM-CTC model with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.


.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.


Data preparation
----------------

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./prepare.sh

The script ``./prepare.sh`` handles the data preparation for you, **automagically**.
All you need to do is to run it.

The data preparation contains several stages, you can use the following two
options:

  - ``--stage``
  - ``--stop-stage``

to control which stage(s) should be run. By default, all stages are executed.


For example,

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./prepare.sh --stage 0 --stop-stage 0

means to run only stage 0.

To run stage 2 to stage 5, use:

.. code-block:: bash

  $ ./prepare.sh --stage 2 --stop-stage 5

We provide the following YouTube video showing how to run ``./prepare.sh``.

.. note::

   To get the latest news of `next-gen Kaldi <https://github.com/k2-fsa>`_, please subscribe
   the following YouTube channel by `Nadira Povey <https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_:

      `<https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_

..  youtube:: ofEIoJL-mGM

Training
--------

Now describing the training of TDNN-LSTM-CTC model, contained in
the `tdnn_lstm_ctc <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/tdnn_lstm_ctc>`_
folder.

The command to run the training part is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ export CUDA_VISIBLE_DEVICES="0,1,2,3"
  $ ./tdnn_lstm_ctc/train.py --world-size 4

By default, it will run ``20`` epochs. Training logs and checkpoints are saved
in ``tdnn_lstm_ctc/exp``.

In ``tdnn_lstm_ctc/exp``, you will find the following files:

  - ``epoch-0.pt``, ``epoch-1.pt``, ..., ``epoch-19.pt``

    These are checkpoint files, containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./tdnn_lstm_ctc/train.py --start-epoch 11

  - ``tensorboard/``

    This folder contains TensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd tdnn_lstm_ctc/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "TDNN LSTM training for librispeech with icefall"

  - ``log/log-train-xxxx``

    It is the detailed training log in text format, same as the one
    you saw printed to the console during training.


To see available training options, you can use:

.. code-block:: bash

  $ ./tdnn_lstm_ctc/train.py --help

Other training options, e.g., learning rate, results dir, etc., are
pre-configured in the function ``get_params()``
in `tdnn_lstm_ctc/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/tdnn_lstm_ctc/train.py>`_.
Normally, you don't need to change them. You can change them by modifying the code, if
you want.

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

The command for decoding is:

.. code-block:: bash

  $ export CUDA_VISIBLE_DEVICES="0"
  $ ./tdnn_lstm_ctc/decode.py

You will see the WER in the output log.

Decoded results are saved in ``tdnn_lstm_ctc/exp``.

.. code-block:: bash

  $ ./tdnn_lstm_ctc/decode.py --help

shows you the available decoding options.

Some commonly used options are:

  - ``--epoch``

    You can select which checkpoint to be used for decoding.
    For instance, ``./tdnn_lstm_ctc/decode.py --epoch 10`` means to use
    ``./tdnn_lstm_ctc/exp/epoch-10.pt`` for decoding.

  - ``--avg``

    It's related to model averaging. It specifies number of checkpoints
    to be averaged. The averaged model is used for decoding.
    For example, the following command:

      .. code-block:: bash

        $ ./tdnn_lstm_ctc/decode.py --epoch 10 --avg 3

    uses the average of ``epoch-8.pt``, ``epoch-9.pt`` and ``epoch-10.pt``
    for decoding.

  - ``--export``

    If it is ``True``, i.e., ``./tdnn_lstm_ctc/decode.py --export 1``, the code
    will save the averaged model to ``tdnn_lstm_ctc/exp/pretrained.pt``.
    See :ref:`tdnn_lstm_ctc use a pre-trained model` for how to use it.


.. _tdnn_lstm_ctc use a pre-trained model:

Pre-trained Model
-----------------

We have uploaded the pre-trained model to
`<https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc>`_.

The following shows you how to use the pre-trained model.


Install kaldifeat
~~~~~~~~~~~~~~~~~

`kaldifeat <https://github.com/csukuangfj/kaldifeat>`_ is used to
extract features for a single sound file or multiple sound files
at the same time.

Please refer to `<https://github.com/csukuangfj/kaldifeat>`_ for installation.

Download the pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ mkdir tmp
  $ cd tmp
  $ git lfs install
  $ git clone https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc

.. CAUTION::

  You have to use ``git lfs`` to download the pre-trained model.

.. CAUTION::

  In order to use this pre-trained model, your k2 version has to be v1.7 or later.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ tree tmp

.. code-block:: bash

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

**File descriptions**:

  - ``data/lang_phone/HLG.pt``

      It is the decoding graph.

  - ``data/lang_phone/tokens.txt``

      It contains tokens and their IDs.

  - ``data/lang_phone/words.txt``

      It contains words and their IDs.

  - ``data/lm/G_4_gram.pt``

      It is a 4-gram LM, useful for LM rescoring.

  - ``exp/pretrained.pt``

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from ``epoch-14.pt`` to ``epoch-19.pt``.
      Note: We have removed optimizer ``state_dict`` to reduce file size.

  - ``test_waves/*.flac``

      It contains some test sound files from LibriSpeech ``test-clean`` dataset.

  - ``test_waves/trans.txt``

      It contains the reference transcripts for the sound files in ``test_waves/``.

The information of the test sound files is listed below:

.. code-block:: bash

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


Inference with a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./tdnn_lstm_ctc/pretrained.py --help

shows the usage information of ``./tdnn_lstm_ctc/pretrained.py``.

To decode with ``1best`` method, we can use:

.. code-block:: bash

  ./tdnn_lstm_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/exp/pretraind.pt \
    --words-file ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/words.txt \
    --HLG ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt \
    ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac \
    ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac \
    ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac

The output is:

.. code-block::

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


To decode with ``whole-lattice-rescoring`` methond, you can use

.. code-block:: bash

  ./tdnn_lstm_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/exp/pretraind.pt \
    --words-file ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/words.txt \
    --HLG ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt \
    --method whole-lattice-rescoring \
    --G ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/data/lm/G_4_gram.pt \
    --ngram-lm-scale 0.8 \
    ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1089-134686-0001.flac \
    ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0001.flac \
    ./tmp/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/1221-135766-0002.flac

The decoding output is:

.. code-block::

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


Colab notebook
--------------

We provide a colab notebook for decoding with pre-trained model.

|librispeech tdnn_lstm_ctc colab notebook|

.. |librispeech tdnn_lstm_ctc colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1-iSfQMp2So-We_Uu49N4AAcMInB72u9z?usp=sharing


**Congratulations!** You have finished the TDNN-LSTM-CTC recipe on librispeech in ``icefall``.
