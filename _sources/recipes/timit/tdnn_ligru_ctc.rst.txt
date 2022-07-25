TDNN-LiGRU-CTC
==============

This tutorial shows you how to run a TDNN-LiGRU-CTC model with the `TIMIT <https://data.deepai.org/timit.zip>`_ dataset.


.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.


Data preparation
----------------

.. code-block:: bash

  $ cd egs/timit/ASR
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

  $ cd egs/timit/ASR
  $ ./prepare.sh --stage 0 --stop-stage 0

means to run only stage 0.

To run stage 2 to stage 5, use:

.. code-block:: bash

  $ ./prepare.sh --stage 2 --stop-stage 5


Training
--------

Now describing the training of TDNN-LiGRU-CTC model, contained in
the `tdnn_ligru_ctc <https://github.com/k2-fsa/icefall/tree/master/egs/timit/ASR/tdnn_ligru_ctc>`_
folder.

.. HINT::

  TIMIT is a very small dataset. So one GPU is enough.

The command to run the training part is:

.. code-block:: bash

  $ cd egs/timit/ASR
  $ export CUDA_VISIBLE_DEVICES="0"
  $ ./tdnn_ligru_ctc/train.py

By default, it will run ``25`` epochs. Training logs and checkpoints are saved
in ``tdnn_ligru_ctc/exp``.

In ``tdnn_ligru_ctc/exp``, you will find the following files:

  - ``epoch-0.pt``, ``epoch-1.pt``, ..., ``epoch-29.pt``

    These are checkpoint files, containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./tdnn_ligru_ctc/train.py --start-epoch 11

  - ``tensorboard/``

    This folder contains TensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd tdnn_ligru_ctc/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "TDNN ligru training for timit with icefall"

  - ``log/log-train-xxxx``

    It is the detailed training log in text format, same as the one
    you saw printed to the console during training.


To see available training options, you can use:

.. code-block:: bash

  $ ./tdnn_ligru_ctc/train.py --help

Other training options, e.g., learning rate, results dir, etc., are
pre-configured in the function ``get_params()``
in `tdnn_ligru_ctc/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/timit/ASR/tdnn_ligru_ctc/train.py>`_.
Normally, you don't need to change them. You can change them by modifying the code, if
you want.

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

The command for decoding is:

.. code-block:: bash

  $ export CUDA_VISIBLE_DEVICES="0"
  $ ./tdnn_ligru_ctc/decode.py

You will see the WER in the output log.

Decoded results are saved in ``tdnn_ligru_ctc/exp``.

.. code-block:: bash

  $ ./tdnn_ligru_ctc/decode.py --help

shows you the available decoding options.

Some commonly used options are:

  - ``--epoch``

    You can select which checkpoint to be used for decoding.
    For instance, ``./tdnn_ligru_ctc/decode.py --epoch 10`` means to use
    ``./tdnn_ligru_ctc/exp/epoch-10.pt`` for decoding.

  - ``--avg``

    It's related to model averaging. It specifies number of checkpoints
    to be averaged. The averaged model is used for decoding.
    For example, the following command:

      .. code-block:: bash

        $ ./tdnn_ligru_ctc/decode.py --epoch 25 --avg 17

    uses the average of ``epoch-9.pt``, ``epoch-10.pt``, ``epoch-11.pt``, 
    ``epoch-12.pt``, ``epoch-13.pt``, ``epoch-14.pt``, ``epoch-15.pt``, 
    ``epoch-16.pt``, ``epoch-17.pt``, ``epoch-18.pt``, ``epoch-19.pt``, 
    ``epoch-20.pt``, ``epoch-21.pt``, ``epoch-22.pt``, ``epoch-23.pt``, 
    ``epoch-24.pt`` and ``epoch-25.pt``
    for decoding.

  - ``--export``

    If it is ``True``, i.e., ``./tdnn_ligru_ctc/decode.py --export 1``, the code
    will save the averaged model to ``tdnn_ligru_ctc/exp/pretrained.pt``.
    See :ref:`tdnn_ligru_ctc use a pre-trained model` for how to use it.


.. _tdnn_ligru_ctc use a pre-trained model:

Pre-trained Model
-----------------

We have uploaded the pre-trained model to
`<https://huggingface.co/luomingshuang/icefall_asr_timit_tdnn_ligru_ctc>`_.

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

  $ cd egs/timit/ASR
  $ mkdir tmp-ligru
  $ cd tmp-ligru
  $ git lfs install
  $ git clone https://huggingface.co/luomingshuang/icefall_asr_timit_tdnn_ligru_ctc

.. CAUTION::

  You have to use ``git lfs`` to download the pre-trained model.

.. CAUTION::

  In order to use this pre-trained model, your k2 version has to be v1.7 or later.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/timit/ASR
  $ tree tmp-ligru

.. code-block:: bash

  tmp-ligru/
  `-- icefall_asr_timit_tdnn_ligru_ctc
      |-- README.md
      |-- data
      |   |-- lang_phone
      |   |   |-- HLG.pt
      |   |   |-- tokens.txt
      |   |   `-- words.txt
      |   `-- lm
      |       `-- G_4_gram.pt
      |-- exp
      |   `-- pretrained_average_9_25.pt
      `-- test_wavs
          |-- FDHC0_SI1559.WAV
          |-- FELC0_SI756.WAV
          |-- FMGD0_SI1564.WAV
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
      checkpoints from ``epoch-9.pt`` to ``epoch-25.pt``.
      Note: We have removed optimizer ``state_dict`` to reduce file size.

  - ``test_waves/*.WAV``

      It contains some test sound files from timit ``TEST`` dataset.

  - ``test_waves/trans.txt``

      It contains the reference transcripts for the sound files in ``test_waves/``.

The information of the test sound files is listed below:

.. code-block:: bash

  $ ffprobe -show_format tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV

  Input #0, nistsphere, from 'tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV':
  Metadata:
    database_id     : TIMIT
    database_version: 1.0
    utterance_id    : dhc0_si1559
    sample_min      : -4176
    sample_max      : 5984
  Duration: 00:00:03.40, bitrate: 258 kb/s
    Stream #0:0: Audio: pcm_s16le, 16000 Hz, 1 channels, s16, 256 kb/s

  $ ffprobe -show_format tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV

  Input #0, nistsphere, from 'tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV':
  Metadata:
    database_id     : TIMIT
    database_version: 1.0
    utterance_id    : elc0_si756
    sample_min      : -1546
    sample_max      : 1989
  Duration: 00:00:04.19, bitrate: 257 kb/s
    Stream #0:0: Audio: pcm_s16le, 16000 Hz, 1 channels, s16, 256 kb/s

  $ ffprobe -show_format tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV

  Input #0, nistsphere, from 'tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV':
  Metadata:
    database_id     : TIMIT
    database_version: 1.0
    utterance_id    : mgd0_si1564
    sample_min      : -7626
    sample_max      : 10573
  Duration: 00:00:04.44, bitrate: 257 kb/s
    Stream #0:0: Audio: pcm_s16le, 16000 Hz, 1 channels, s16, 256 kb/s


Inference with a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/timit/ASR
  $ ./tdnn_ligru_ctc/pretrained.py --help

shows the usage information of ``./tdnn_ligru_ctc/pretrained.py``.

To decode with ``1best`` method, we can use:

.. code-block:: bash

  ./tdnn_ligru_ctc/pretrained.py 
    --method 1best
    --checkpoint ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/exp/pretrained_average_9_25.pt 
    --words-file ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lang_phone/words.txt 
    --HLG ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lang_phone/HLG.pt 
    ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV 
    ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV 
    ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV

The output is:

.. code-block::

  2021-11-08 20:41:33,660 INFO [pretrained.py:169] device: cuda:0
  2021-11-08 20:41:33,660 INFO [pretrained.py:171] Creating model
  2021-11-08 20:41:38,680 INFO [pretrained.py:183] Loading HLG from ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lang_phone/HLG.pt
  2021-11-08 20:41:38,695 INFO [pretrained.py:200] Constructing Fbank computer
  2021-11-08 20:41:38,697 INFO [pretrained.py:210] Reading sound files: ['./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV', './tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV', './tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV']
  2021-11-08 20:41:38,704 INFO [pretrained.py:216] Decoding started
  2021-11-08 20:41:39,819 INFO [pretrained.py:246] Use HLG decoding
  2021-11-08 20:41:39,829 INFO [pretrained.py:267] 
  ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV:
  sil dh ih sh uw ah l iy v iy z ih sil p r aa sil k s ih m ey dx ih sil d w uh dx ih w ih s f iy l ih ng w ih th ih n ih m s eh l f sil jh

  ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV:
  sil m ih sil t ih r iy s sil s er r ih m ih sil m aa l ih sil k l ey sil r eh sil d w ay sil d aa r sil b ah f sil jh

  ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV:
  sil hh ah z sil b ih sil g r iy w ah z sil d aw n ih sil b ay s sil n ey sil w eh l f eh n s ih z eh n dh eh r w er sil g r ey z ih ng sil k ae dx l sil


  2021-11-08 20:41:39,829 INFO [pretrained.py:269] Decoding Done


To decode with ``whole-lattice-rescoring`` methond, you can use

.. code-block:: bash

  ./tdnn_ligru_ctc/pretrained.py \
    --method whole-lattice-rescoring \
    --checkpoint ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/exp/pretrained_average_9_25.pt \
    --words-file ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lang_phone/words.txt \
    --HLG ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lang_phone/HLG.pt \
    --G ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lm/G_4_gram.pt \
    --ngram-lm-scale 0.1 \
    ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV 
    ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV 
    ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV

The decoding output is:

.. code-block::

  2021-11-08 20:37:50,693 INFO [pretrained.py:169] device: cuda:0
  2021-11-08 20:37:50,693 INFO [pretrained.py:171] Creating model
  2021-11-08 20:37:54,693 INFO [pretrained.py:183] Loading HLG from ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lang_phone/HLG.pt
  2021-11-08 20:37:54,705 INFO [pretrained.py:191] Loading G from ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/data/lm/G_4_gram.pt
  2021-11-08 20:37:54,714 INFO [pretrained.py:200] Constructing Fbank computer
  2021-11-08 20:37:54,715 INFO [pretrained.py:210] Reading sound files: ['./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV', './tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV', './tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV']
  2021-11-08 20:37:54,720 INFO [pretrained.py:216] Decoding started
  2021-11-08 20:37:55,808 INFO [pretrained.py:251] Use HLG decoding + LM rescoring
  2021-11-08 20:37:56,348 INFO [pretrained.py:267] 
  ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FDHC0_SI1559.WAV:
  sil dh ih sh uw ah l iy v iy z ah sil p r aa sil k s ih m ey dx ih sil d w uh dx iy w ih s f iy l iy ng w ih th ih n ih m s eh l f sil jh

  ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FELC0_SI756.WAV:
  sil m ih sil t ih r iy l s sil s er r eh m ih sil m aa l ih ng sil k l ey sil r eh sil d w ay sil d aa r sil b ah f sil jh ch

  ./tmp-ligru/icefall_asr_timit_tdnn_ligru_ctc/test_waves/FMGD0_SI1564.WAV:
  sil hh ah z sil b ih n sil g r iy w ah z sil b aw n ih sil b ay s sil n ey sil w er l f eh n s ih z eh n dh eh r w er sil g r ey z ih ng sil k ae dx l sil


  2021-11-08 20:37:56,348 INFO [pretrained.py:269] Decoding Done


Colab notebook
--------------

We provide a colab notebook for decoding with pre-trained model.

|timit tdnn_ligru_ctc colab notebook|

.. |timit tdnn_ligru_ctc colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/11IT-k4HQIgQngXz1uvWsEYktjqQt7Tmb


**Congratulations!** You have finished the TDNN-LiGRU-CTC recipe on timit in ``icefall``.
