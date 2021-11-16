Confromer CTC
=============

This tutorial shows you how to run a conformer ctc model
with the `Aishell <https://www.openslr.org/33>`_ dataset.


.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe.

In this tutorial, you will learn:

  - (1) How to prepare data for training and decoding
  - (2) How to start the training, either with a single GPU or multiple GPUs
  - (3) How to do decoding after training, with 1best and attention decoder rescoring
  - (4) How to use a pre-trained model, provided by us

Data preparation
----------------

.. code-block:: bash

  $ cd egs/aishell/ASR
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

  $ cd egs/aishell/ASR
  $ ./prepare.sh --stage 0 --stop-stage 0

means to run only stage 0.

To run stage 2 to stage 5, use:

.. code-block:: bash

  $ ./prepare.sh --stage 2 --stop-stage 5

.. HINT::

  If you have pre-downloaded the `Aishell <https://www.openslr.org/33>`_
  dataset and the `musan <http://www.openslr.org/17/>`_ dataset, say,
  they are saved in ``/tmp/aishell`` and ``/tmp/musan``, you can modify
  the ``dl_dir`` variable in ``./prepare.sh`` to point to ``/tmp`` so that
  ``./prepare.sh`` won't re-download them.

.. HINT::

  A 3-gram language model will be downloaded from huggingface, we assume you have
  intalled and initialized ``git-lfs``. If not, you could install ``git-lfs`` by

  .. code-block:: bash

    $ sudo apt-get install git-lfs
    $ git-lfs install

  If you don't have the ``sudo`` permission, you could download the
  `git-lfs binary <https://github.com/git-lfs/git-lfs/releases>`_ here, then add it to you ``PATH``.

.. NOTE::

  All generated files by ``./prepare.sh``, e.g., features, lexicon, etc,
  are saved in ``./data`` directory.


Training
--------

Configurable options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/train.py --help

shows you the training options that can be passed from the commandline.
The following options are used quite often:

  - ``--exp-dir``

    The experiment folder to save logs and model checkpoints,
    default ``./conformer_ctc/exp``.

  - ``--num-epochs``

    It is the number of epochs to train. For instance,
    ``./conformer_ctc/train.py --num-epochs 30`` trains for 30 epochs
    and generates ``epoch-0.pt``, ``epoch-1.pt``, ..., ``epoch-29.pt``
    in the folder set with ``--exp-dir``.

  - ``--start-epoch``

    It's used to resume training.
    ``./conformer_ctc/train.py --start-epoch 10`` loads the
    checkpoint ``./conformer_ctc/exp/epoch-9.pt`` and starts
    training from epoch 10, based on the state from epoch 9.

  - ``--world-size``

    It is used for multi-GPU single-machine DDP training.

      - (a) If it is 1, then no DDP training is used.

      - (b) If it is 2, then GPU 0 and GPU 1 are used for DDP training.

    The following shows some use cases with it.

      **Use case 1**: You have 4 GPUs, but you only want to use GPU 0 and
      GPU 2 for training. You can do the following:

        .. code-block:: bash

          $ cd egs/aishell/ASR
          $ export CUDA_VISIBLE_DEVICES="0,2"
          $ ./conformer_ctc/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/aishell/ASR
          $ ./conformer_ctc/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/aishell/ASR
          $ export CUDA_VISIBLE_DEVICES="3"
          $ ./conformer_ctc/train.py --world-size 1

    .. CAUTION::

      Only multi-GPU single-machine DDP training is implemented at present.
      Multi-GPU multi-machine DDP training will be added later.

  - ``--max-duration``

    It specifies the number of seconds over all utterances in a
    batch, before **padding**.
    If you encounter CUDA OOM, please reduce it. For instance, if
    your are using V100 NVIDIA GPU, we recommend you to set it to ``200``.

    .. HINT::

      Due to padding, the number of seconds of all utterances in a
      batch will usually be larger than ``--max-duration``.

      A larger value for ``--max-duration`` may cause OOM during training,
      while a smaller value may increase the training time. You have to
      tune it.


Pre-configured options
~~~~~~~~~~~~~~~~~~~~~~

There are some training options, e.g., weight decay,
number of warmup steps, etc,
that are not passed from the commandline.
They are pre-configured by the function ``get_params()`` in
`conformer_ctc/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/conformer_ctc/train.py>`_

You don't need to change these pre-configured parameters. If you really need to change
them, please modify ``./conformer_ctc/train.py`` directly.


.. CAUTION::

  The training set is perturbed by speed with two factors: 0.9 and 1.1.
  Each epoch actually processes ``3x150 == 450`` hours of data.


Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in the folder set by ``--exp-dir``
(default ``conformer_ctc/exp``). You will find the following files in that directory:

  - ``epoch-0.pt``, ``epoch-1.pt``, ...

    These are checkpoint files, containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./conformer_ctc/train.py --start-epoch 11

  - ``tensorboard/``

    This folder contains TensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd conformer_ctc/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "Conformer CTC training for Aishell with icefall"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/WE1DocDqRRCOSAgmGyClhg/

        [2021-11-16T10:51:46] Started scanning logdir.
        [2021-11-16T10:52:32] Total uploaded: 111606 scalars, 0 tensors, 0 binary objects
        Listening for new data in logdir...

    Note there is a URL in the above output, click it and you will see
    the following screenshot:

      .. figure:: images/aishell-conformer-ctc-tensorboard-log.jpg
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/WE1DocDqRRCOSAgmGyClhg/

         TensorBoard screenshot.

  - ``log/log-train-xxxx``

    It is the detailed training log in text format, same as the one
    you saw printed to the console during training.

Usage examples
~~~~~~~~~~~~~~

The following shows typical use cases:

**Case 1**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/train.py --max-duration 200

It uses ``--max-duration`` of 200 to avoid OOM.


**Case 2**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ export CUDA_VISIBLE_DEVICES="0,3"
  $ ./conformer_ctc/train.py --world-size 2

It uses GPU 0 and GPU 3 for DDP training.

**Case 3**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/train.py --num-epochs 10 --start-epoch 3

It loads checkpoint ``./conformer_ctc/exp/epoch-2.pt`` and starts
training from epoch 3. Also, it trains for 10 epochs.

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/decode.py --help

shows the options for decoding.

The commonly used options are:

  - ``--method``

    This specifies the decoding method.

    The following command uses attention decoder for rescoring:

    .. code-block::

      $ cd egs/aishell/ASR
      $ ./conformer_ctc/decode.py --method attention-decoder --max-duration 30 --nbest-scale 0.5

  - ``--nbest-scale``

    It is used to scale down lattice scores so that there are more unique
    paths for rescoring.

  - ``--max-duration``

    It has the same meaning as the one during training. A larger
    value may cause OOM.

Pre-trained Model
-----------------

We have uploaded a pre-trained model to
`<https://huggingface.co/pkufool/icefall_asr_aishell_conformer_ctc>`_.

We describe how to use the pre-trained model to transcribe a sound file or
multiple sound files in the following.

Install kaldifeat
~~~~~~~~~~~~~~~~~

`kaldifeat <https://github.com/csukuangfj/kaldifeat>`_ is used to
extract features for a single sound file or multiple sound files
at the same time.

Please refer to `<https://github.com/csukuangfj/kaldifeat>`_ for installation.

Download the pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following commands describe how to download the pre-trained model:

.. code-block::

  $ cd egs/aishell/ASR
  $ mkdir tmp
  $ cd tmp
  $ git lfs install
  $ git clone https://huggingface.co/pkufool/icefall_asr_aishell_conformer_ctc

.. CAUTION::

  You have to use ``git lfs`` to download the pre-trained model.

.. CAUTION::

  In order to use this pre-trained model, your k2 version has to be v1.7 or later.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ tree tmp

.. code-block:: bash

  tmp/
  `-- icefall_asr_aishell_conformer_ctc
      |-- README.md
      |-- data
      |   `-- lang_char
      |       |-- HLG.pt
      |       |-- tokens.txt
      |       `-- words.txt
      |-- exp
      |   `-- pretrained.pt
      `-- test_waves
          |-- BAC009S0764W0121.wav
          |-- BAC009S0764W0122.wav
          |-- BAC009S0764W0123.wav
          `-- trans.txt

  5 directories, 9 files

**File descriptions**:

  - ``data/lang_char/HLG.pt``

      It is the decoding graph.

  - ``data/lang_char/tokens.txt``

      It contains tokens and their IDs.
      Provided only for convenience so that you can look up the SOS/EOS ID easily.

  - ``data/lang_char/words.txt``

      It contains words and their IDs.

  - ``exp/pretrained.pt``

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from ``epoch-25.pt`` to ``epoch-84.pt``.
      Note: We have removed optimizer ``state_dict`` to reduce file size.

  - ``test_waves/*.wav``

      It contains some test sound files from Aishell ``test`` dataset.

  - ``test_waves/trans.txt``

      It contains the reference transcripts for the sound files in `test_waves/`.

The information of the test sound files is listed below:

.. code-block:: bash

  $ soxi tmp/icefall_asr_aishell_conformer_ctc/test_wavs/*.wav

  Input File     : 'tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0121.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.20 = 67263 samples ~ 315.295 CDDA sectors
  File Size      : 135k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0122.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.12 = 65840 samples ~ 308.625 CDDA sectors
  File Size      : 132k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0123.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.00 = 64000 samples ~ 300 CDDA sectors
  File Size      : 128k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM

  Total Duration of 3 files: 00:00:12.32

Usage
~~~~~

.. code-block::

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/pretrained.py --help

displays the help information.

It supports two decoding methods:

  - HLG decoding
  - HLG + attention decoder rescoring

HLG decoding
^^^^^^^^^^^^

HLG decoding uses the best path of the decoding lattice as the decoding result.

The command to run HLG decoding is:

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_aishell_conformer_ctc/exp/pretrained.pt \
    --words-file ./tmp/icefall_asr_aishell_conformer_ctc/data/lang_char/words.txt \
    --HLG ./tmp/icefall_asr_aishell_conformer_ctc/data/lang_char/HLG.pt \
    --method 1best \
    ./tmp/icefall_asr_aishell_conformer_ctc/test_wavs/BAC009S0764W0121.wav \
    ./tmp/icefall_asr_aishell_conformer_ctc/test_wavs/BAC009S0764W0122.wav \
    ./tmp/icefall_asr_aishell_conformer_ctc/test_wavs/BAC009S0764W0123.wav

The output is given below:

.. code-block::

  2021-09-13 10:46:59,842 INFO [pretrained.py:219] device: cuda:0
  2021-09-13 10:46:59,842 INFO [pretrained.py:221] Creating model
  2021-09-13 10:47:54,682 INFO [pretrained.py:238] Loading HLG from ./tmp/icefall_asr_aishell_conformer_ctc/data/lang_char/HLG.pt
  2021-09-13 10:48:46,111 INFO [pretrained.py:245] Constructing Fbank computer
  2021-09-13 10:48:46,113 INFO [pretrained.py:255] Reading sound files: ['./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0121.wav', './tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0122.wav', './tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0123.wav']
  2021-09-13 10:48:46,368 INFO [pretrained.py:262] Decoding started
  2021-09-13 10:48:46,847 INFO [pretrained.py:291] Use HLG decoding
  2021-09-13 10:48:47,176 INFO [pretrained.py:322]
  ./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0121.wav:
  甚至 出现 交易 几乎 停止 的 情况

  ./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0122.wav:
  一二 线 城市 虽然 也 处于 调整 中

  ./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0123.wav:
  但 因为 聚集 了 过多 公共 资源


  2021-09-13 10:48:47,177 INFO [pretrained.py:324] Decoding Done

HLG decoding + attention decoder rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It extracts n paths from the lattice, recores the extracted paths with
an attention decoder. The path with the highest score is the decoding result.

The command to run HLG decoding + attention decoder rescoring is:

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./conformer_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_aishell_conformer_ctc/exp/pretrained.pt \
    --words-file ./tmp/icefall_asr_aishell_conformer_ctc/data/lang_char/words.txt \
    --HLG ./tmp/icefall_asr_aishell_conformer_ctc/data/lang_char/HLG.pt \
    --method attention-decoder \
    ./tmp/icefall_asr_aishell_conformer_ctc/test_wavs/BAC009S0764W0121.wav \
    ./tmp/icefall_asr_aishell_conformer_ctc/test_wavs/BAC009S0764W0122.wav \
    ./tmp/icefall_asr_aishell_conformer_ctc/test_wavs/BAC009S0764W0123.wav

The output is below:

.. code-block::

  2021-09-13 11:02:15,852 INFO [pretrained.py:219] device: cuda:0
  2021-09-13 11:02:15,852 INFO [pretrained.py:221] Creating model
  2021-09-13 11:02:22,292 INFO [pretrained.py:238] Loading HLG from ./tmp/icefall_asr_aishell_conformer_ctc/data/lang_char/HLG.pt
  2021-09-13 11:02:27,060 INFO [pretrained.py:245] Constructing Fbank computer
  2021-09-13 11:02:27,062 INFO [pretrained.py:255] Reading sound files: ['./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0121.wav', './tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0122.wav', './tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0123.wav']
  2021-09-13 11:02:27,129 INFO [pretrained.py:261] Decoding started
  2021-09-13 11:02:27,241 INFO [pretrained.py:295] Use HLG + attention decoder rescoring
  2021-09-13 11:02:27,823 INFO [pretrained.py:318]
  ./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0121.wav:
  甚至 出现 交易 几乎 停止 的 情况

  ./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0122.wav:
  一二 线 城市 虽然 也 处于 调整 中

  ./tmp/icefall_asr_aishell_conformer_ctc/test_waves/BAC009S0764W0123.wav:
  但 因为 聚集 了 过多 公共 资源


  2021-09-13 11:02:27,823 INFO [pretrained.py:320] Decoding Done

Colab notebook
--------------

We do provide a colab notebook for this recipe showing how to use a pre-trained model.

|aishell asr conformer ctc colab notebook|

.. |aishell asr conformer ctc colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1WnG17io5HEZ0Gn_cnh_VzK5QYOoiiklC

.. HINT::

  Due to limited memory provided by Colab, you have to upgrade to Colab Pro to
  run ``HLG decoding + attention decoder rescoring``.
  Otherwise, you can only run ``HLG decoding`` with Colab.

**Congratulations!** You have finished the aishell ASR recipe with
conformer CTC models in ``icefall``.
