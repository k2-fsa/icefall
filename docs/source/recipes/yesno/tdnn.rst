TDNN-CTC
========

This page shows you how to run the `yesno <https://www.openslr.org/1>`_ recipe. It contains:

  - (1) Prepare data for training
  - (2) Train a TDNN model

    - (a) View text format logs and visualize TensorBoard logs
    - (b) Select device type, i.e., CPU and GPU, for training
    - (c) Change training options
    - (d) Resume training from a checkpoint

  - (3) Decode with a trained model

    - (a) Select a checkpoint for decoding
    - (b) Model averaging

  - (4) Colab notebook

    - (a) It shows you step by step how to setup the environment, how to do training,
          and how to do decoding
    - (b) How to use a pre-trained model

  - (5) Inference with a pre-trained model

    - (a) Download a pre-trained model, provided by us
    - (b) Decode a single sound file with a pre-trained model
    - (c) Decode multiple sound files at the same time

It does **NOT** show you:

  - (1) How to train with multiple GPUs

    The ``yesno`` dataset is so small that CPU is more than enough
    for training as well as for decoding.

  - (2) How to use LM rescoring for decoding

    The dataset does not have an LM for rescoring.

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  You **don't** need a **GPU** to run this recipe. It can be run on a **CPU**.
  The training part takes less than 30 **seconds** on a CPU and you will get
  the following WER at the end::

    [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]

Data preparation
----------------

.. code-block:: bash

  $ cd egs/yesno/ASR
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

  $ cd egs/yesno/ASR
  $ ./prepare.sh --stage 0 --stop-stage 0

means to run only stage 0.

To run stage 2 to stage 5, use:

.. code-block:: bash

  $ ./prepare.sh --stage 2 --stop-stage 5


Training
--------

We provide only a TDNN model, contained in
the `tdnn <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn>`_
folder, for ``yesno``.

The command to run the training part is:

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ export CUDA_VISIBLE_DEVICES=""
  $ ./tdnn/train.py

By default, it will run ``15`` epochs. Training logs and checkpoints are saved
in ``tdnn/exp``.

In ``tdnn/exp``, you will find the following files:

  - ``epoch-0.pt``, ``epoch-1.pt``, ...

    These are checkpoint files, containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./tdnn/train.py --start-epoch 11

  - ``tensorboard/``

    This folder contains TensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd tdnn/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "TDNN training for yesno with icefall"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/yKUbhb5wRmOSXYkId1z9eg/

        [2021-08-23T23:49:41] Started scanning logdir.
        [2021-08-23T23:49:42] Total uploaded: 135 scalars, 0 tensors, 0 binary objects
        Listening for new data in logdir...

    Note there is a URL in the above output, click it and you will see
    the following screenshot:

      .. figure:: images/tdnn-tensorboard-log.png
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/yKUbhb5wRmOSXYkId1z9eg/

         TensorBoard screenshot.

  - ``log/log-train-xxxx``

    It is the detailed training log in text format, same as the one
    you saw printed to the console during training.



.. NOTE::

  By default, ``./tdnn/train.py`` uses GPU 0 for training if GPUs are available.
  If you have two GPUs, say, GPU 0 and GPU 1, and you want to use GPU 1 for
  training, you can run:

    .. code-block:: bash

      $ export CUDA_VISIBLE_DEVICES="1"
      $ ./tdnn/train.py

  Since the ``yesno`` dataset is very small, containing only 30 sound files
  for training, and the model in use is also very small, we use:

    .. code-block:: bash

      $ export CUDA_VISIBLE_DEVICES=""

  so that ``./tdnn/train.py`` uses CPU during training.

  If you don't have GPUs, then you don't need to
  run ``export CUDA_VISIBLE_DEVICES=""``.

To see available training options, you can use:

.. code-block:: bash

  $ ./tdnn/train.py --help

Other training options, e.g., learning rate, results dir, etc., are
pre-configured in the function ``get_params()``
in `tdnn/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/yesno/ASR/tdnn/train.py>`_.
Normally, you don't need to change them. You can change them by modifying the code, if
you want.

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

The command for decoding is:

.. code-block:: bash

  $ export CUDA_VISIBLE_DEVICES=""
  $ ./tdnn/decode.py

You will see the WER in the output log.

Decoded results are saved in ``tdnn/exp``.

.. code-block:: bash

  $ ./tdnn/decode.py --help

shows you the available decoding options.

Some commonly used options are:

  - ``--epoch``

    You can select which checkpoint to be used for decoding.
    For instance, ``./tdnn/decode.py --epoch 10`` means to use
    ``./tdnn/exp/epoch-10.pt`` for decoding.

  - ``--avg``

    It's related to model averaging. It specifies number of checkpoints
    to be averaged. The averaged model is used for decoding.
    For example, the following command:

      .. code-block:: bash

        $ ./tdnn/decode.py --epoch 10 --avg 3

    uses the average of ``epoch-8.pt``, ``epoch-9.pt`` and ``epoch-10.pt``
    for decoding.

  - ``--export``

    If it is ``True``, i.e., ``./tdnn/decode.py --export 1``, the code
    will save the averaged model to ``tdnn/exp/pretrained.pt``.
    See :ref:`yesno use a pre-trained model` for how to use it.


.. _yesno use a pre-trained model:

Pre-trained Model
-----------------

We have uploaded the pre-trained model to
`<https://huggingface.co/csukuangfj/icefall_asr_yesno_tdnn>`_.

The following shows you how to use the pre-trained model.

Download the pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ mkdir tmp
  $ cd tmp
  $ git lfs install
  $ git clone https://huggingface.co/csukuangfj/icefall_asr_yesno_tdnn

.. CAUTION::

  You have to use ``git lfs`` to download the pre-trained model.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ tree tmp

.. code-block:: bash

  tmp/
  `-- icefall_asr_yesno_tdnn
      |-- README.md
      |-- lang_phone
      |   |-- HLG.pt
      |   |-- L.pt
      |   |-- L_disambig.pt
      |   |-- Linv.pt
      |   |-- lexicon.txt
      |   |-- lexicon_disambig.txt
      |   |-- tokens.txt
      |   `-- words.txt
      |-- lm
      |   |-- G.arpa
      |   `-- G.fst.txt
      |-- pretrained.pt
      `-- test_waves
          |-- 0_0_0_1_0_0_0_1.wav
          |-- 0_0_1_0_0_0_1_0.wav
          |-- 0_0_1_0_0_1_1_1.wav
          |-- 0_0_1_0_1_0_0_1.wav
          |-- 0_0_1_1_0_0_0_1.wav
          |-- 0_0_1_1_0_1_1_0.wav
          |-- 0_0_1_1_1_0_0_0.wav
          |-- 0_0_1_1_1_1_0_0.wav
          |-- 0_1_0_0_0_1_0_0.wav
          |-- 0_1_0_0_1_0_1_0.wav
          |-- 0_1_0_1_0_0_0_0.wav
          |-- 0_1_0_1_1_1_0_0.wav
          |-- 0_1_1_0_0_1_1_1.wav
          |-- 0_1_1_1_0_0_1_0.wav
          |-- 0_1_1_1_1_0_1_0.wav
          |-- 1_0_0_0_0_0_0_0.wav
          |-- 1_0_0_0_0_0_1_1.wav
          |-- 1_0_0_1_0_1_1_1.wav
          |-- 1_0_1_1_0_1_1_1.wav
          |-- 1_0_1_1_1_1_0_1.wav
          |-- 1_1_0_0_0_1_1_1.wav
          |-- 1_1_0_0_1_0_1_1.wav
          |-- 1_1_0_1_0_1_0_0.wav
          |-- 1_1_0_1_1_0_0_1.wav
          |-- 1_1_0_1_1_1_1_0.wav
          |-- 1_1_1_0_0_1_0_1.wav
          |-- 1_1_1_0_1_0_1_0.wav
          |-- 1_1_1_1_0_0_1_0.wav
          |-- 1_1_1_1_1_0_0_0.wav
          `-- 1_1_1_1_1_1_1_1.wav

  4 directories, 42 files

.. code-block:: bash

  $ soxi tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav

  Input File     : 'tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav'
  Channels       : 1
  Sample Rate    : 8000
  Precision      : 16-bit
  Duration       : 00:00:06.76 = 54080 samples ~ 507 CDDA sectors
  File Size      : 108k
  Bit Rate       : 128k
  Sample Encoding: 16-bit Signed Integer PCM

- ``0_0_1_0_1_0_0_1.wav``

    0 means No; 1 means Yes. No and Yes are not in English,
    but in `Hebrew <https://en.wikipedia.org/wiki/Hebrew_language>`_.
    So this file contains ``NO NO YES NO YES NO NO YES``.

Download kaldifeat
~~~~~~~~~~~~~~~~~~

`kaldifeat <https://github.com/csukuangfj/kaldifeat>`_ is used for extracting
features from a single or multiple sound files. Please refer to
`<https://github.com/csukuangfj/kaldifeat>`_ to install ``kaldifeat`` first.

Inference with a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ ./tdnn/pretrained.py --help

shows the usage information of ``./tdnn/pretrained.py``.

To decode a single file, we can use:

.. code-block:: bash

  ./tdnn/pretrained.py \
    --checkpoint ./tmp/icefall_asr_yesno_tdnn/pretrained.pt \
    --words-file ./tmp/icefall_asr_yesno_tdnn/lang_phone/words.txt \
    --HLG ./tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt \
    ./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav

The output is:

.. code-block::

  2021-08-24 12:22:51,621 INFO [pretrained.py:119] {'feature_dim': 23, 'num_classes': 4, 'sample_rate': 8000, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './tmp/icefall_asr_yesno_tdnn/pretrained.pt', 'words_file': './tmp/icefall_asr_yesno_tdnn/lang_phone/words.txt', 'HLG': './tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt', 'sound_files': ['./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav']}
  2021-08-24 12:22:51,645 INFO [pretrained.py:125] device: cpu
  2021-08-24 12:22:51,645 INFO [pretrained.py:127] Creating model
  2021-08-24 12:22:51,650 INFO [pretrained.py:139] Loading HLG from ./tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt
  2021-08-24 12:22:51,651 INFO [pretrained.py:143] Constructing Fbank computer
  2021-08-24 12:22:51,652 INFO [pretrained.py:153] Reading sound files: ['./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav']
  2021-08-24 12:22:51,684 INFO [pretrained.py:159] Decoding started
  2021-08-24 12:22:51,708 INFO [pretrained.py:198]
  ./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav:
  NO NO YES NO YES NO NO YES


  2021-08-24 12:22:51,708 INFO [pretrained.py:200] Decoding Done

You can see that for the sound file ``0_0_1_0_1_0_0_1.wav``, the decoding result is
``NO NO YES NO YES NO NO YES``.

To decode **multiple** files at the same time, you can use

.. code-block:: bash

  ./tdnn/pretrained.py \
    --checkpoint ./tmp/icefall_asr_yesno_tdnn/pretrained.pt \
    --words-file ./tmp/icefall_asr_yesno_tdnn/lang_phone/words.txt \
    --HLG ./tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt \
    ./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav \
    ./tmp/icefall_asr_yesno_tdnn/test_waves/1_0_1_1_0_1_1_1.wav

The decoding output is:

.. code-block::

  2021-08-24 12:25:20,159 INFO [pretrained.py:119] {'feature_dim': 23, 'num_classes': 4, 'sample_rate': 8000, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './tmp/icefall_asr_yesno_tdnn/pretrained.pt', 'words_file': './tmp/icefall_asr_yesno_tdnn/lang_phone/words.txt', 'HLG': './tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt', 'sound_files': ['./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav', './tmp/icefall_asr_yesno_tdnn/test_waves/1_0_1_1_0_1_1_1.wav']}
  2021-08-24 12:25:20,181 INFO [pretrained.py:125] device: cpu
  2021-08-24 12:25:20,181 INFO [pretrained.py:127] Creating model
  2021-08-24 12:25:20,185 INFO [pretrained.py:139] Loading HLG from ./tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt
  2021-08-24 12:25:20,186 INFO [pretrained.py:143] Constructing Fbank computer
  2021-08-24 12:25:20,187 INFO [pretrained.py:153] Reading sound files: ['./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav',
  './tmp/icefall_asr_yesno_tdnn/test_waves/1_0_1_1_0_1_1_1.wav']
  2021-08-24 12:25:20,213 INFO [pretrained.py:159] Decoding started
  2021-08-24 12:25:20,287 INFO [pretrained.py:198]
  ./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav:
  NO NO YES NO YES NO NO YES

  ./tmp/icefall_asr_yesno_tdnn/test_waves/1_0_1_1_0_1_1_1.wav:
  YES NO YES YES NO YES YES YES

  2021-08-24 12:25:20,287 INFO [pretrained.py:200] Decoding Done

You can see again that it decodes correctly.

Colab notebook
--------------

We do provide a colab notebook for this recipe.

|yesno colab notebook|

.. |yesno colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing


**Congratulations!** You have finished the simplest speech recognition recipe in ``icefall``.
