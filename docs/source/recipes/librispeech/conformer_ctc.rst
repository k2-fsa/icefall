Confromer CTC
=============

This tutorial shows you how to run a conformer ctc model
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.


.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe.

In this tutorial, you will learn:

  - (1) How to prepare data for training and decoding
  - (2) How to start the training, either with a single GPU or multiple GPUs
  - (3) How to do decoding after training, with n-gram LM rescoring and attention decoder rescoring
  - (4) How to use a pre-trained model, provided by us

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

.. HINT::

  If you have pre-downloaded the `LibriSpeech <https://www.openslr.org/12>`_
  dataset and the `musan <http://www.openslr.org/17/>`_ dataset, say,
  they are saved in ``/tmp/LibriSpeech`` and ``/tmp/musan``, you can modify
  the ``dl_dir`` variable in ``./prepare.sh`` to point to ``/tmp`` so that
  ``./prepare.sh`` won't re-download them.

.. NOTE::

  All generated files by ``./prepare.sh``, e.g., features, lexicon, etc,
  are saved in ``./data`` directory.


Training
--------

Configurable options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/train.py --help

shows you the training options that can be passed from the commandline.
The following options are used quite often:

  - ``--full-libri``

    If it's True, the training part uses all the training data, i.e.,
    960 hours. Otherwise, the training part uses only the subset
    ``train-clean-100``, which has 100 hours of training data.

    .. CAUTION::

      The training set is perturbed by speed with two factors: 0.9 and 1.1.
      If ``--full-libri`` is True, each epoch actually processes
      ``3x960 == 2880`` hours of data.

  - ``--num-epochs``

    It is the number of epochs to train. For instance,
    ``./conformer_ctc/train.py --num-epochs 30`` trains for 30 epochs
    and generates ``epoch-0.pt``, ``epoch-1.pt``, ..., ``epoch-29.pt``
    in the folder ``./conformer_ctc/exp``.

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

          $ cd egs/librispeech/ASR
          $ export CUDA_VISIBLE_DEVICES="0,2"
          $ ./conformer_ctc/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ ./conformer_ctc/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
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
number of warmup steps, results dir, etc,
that are not passed from the commandline.
They are pre-configured by the function ``get_params()`` in
`conformer_ctc/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/train.py>`_

You don't need to change these pre-configured parameters. If you really need to change
them, please modify ``./conformer_ctc/train.py`` directly.


Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in ``conformer_ctc/exp``.
You will find the following files in that directory:

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
        $ tensorboard dev upload --logdir . --description "Conformer CTC training for LibriSpeech with icefall"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/lzGnETjwRxC3yghNMd4kPw/

        [2021-08-24T16:42:43] Started scanning logdir.
        Uploading 4540 scalars...

    Note there is a URL in the above output, click it and you will see
    the following screenshot:

      .. figure:: images/librispeech-conformer-ctc-tensorboard-log.png
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/lzGnETjwRxC3yghNMd4kPw/

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

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/train.py --max-duration 200 --full-libri 0

It uses ``--max-duration`` of 200 to avoid OOM.  Also, it uses only
a subset of the LibriSpeech data for training.


**Case 2**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ export CUDA_VISIBLE_DEVICES="0,3"
  $ ./conformer_ctc/train.py --world-size 2

It uses GPU 0 and GPU 3 for DDP training.

**Case 3**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/train.py --num-epochs 10 --start-epoch 3

It loads checkpoint ``./conformer_ctc/exp/epoch-2.pt`` and starts
training from epoch 3. Also, it trains for 10 epochs.

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/decode.py --help

shows the options for decoding.

The commonly used options are:

  - ``--method``

    This specifies the decoding method. This script support seven decoding methods. As for ctc decoding, 
    it uses a sentence piece model to convert word pieces to words. And it needs neither a lexicon
    nor an n-gram LM.
    
    For example, the following command uses CTC topology for rescoring:
    
    .. code-block::

      $ cd egs/librispeech/ASR
      $ ./conformer_ctc/decode.py --method ctc-decoding --max-duration 300 --bucketing-sampler False

    And the following command uses attention decoder for rescoring:

    .. code-block::

      $ cd egs/librispeech/ASR
      $ ./conformer_ctc/decode.py --method attention-decoder --max-duration 30 --nbest-scale 0.5

  - ``--nbest-scale``

    It is used to scale down lattice scores so that there are more unique
    paths for rescoring.

  - ``--max-duration``

    It has the same meaning as the one during training. A larger
    value may cause OOM.
    
  - ``--bucketing-sampler``

    When enabled, the batches will come from buckets of similar duration (saves padding frames). 

Here are some results for reference based on CTC decoding when set vocab size as 500:

Usage:

.. code-block:: bash
   $ cd egs/librispeech/ASR
   $ ./conformer_ctc/decode.py \
      --epoch 25 \
      --avg 1 \
      --max-duration 300 \
      --bucketing-sampler 0 \
      --full-libri 0 \
      --exp-dir conformer_ctc/exp \
      --lang-dir data/lang_bpe_500 \
      --method ctc-decoding
      
The output is given below:

.. code-block:: bash

  2021-09-26 12:44:31,033 INFO [decode.py:537] Decoding started
  2021-09-26 12:44:31,033 INFO [decode.py:538] 
  {'lm_dir': PosixPath('data/lm'), 'subsampling_factor': 4, 'vgg_frontend': False, 'use_feat_batchnorm': True, 
  'feature_dim': 80, 'nhead': 8, 'attention_dim': 512, 'num_decoder_layers': 6, 'search_beam': 20, 'output_beam': 8,
  'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 
  'epoch': 25, 'avg': 1, 'method': 'ctc-decoding', 'num_paths': 100, 'nbest_scale': 0.5, 
  'export': False, 'exp_dir': PosixPath('conformer_ctc/exp'), 'lang_dir': PosixPath('data/lang_bpe_500'), 'full_libri': False, 
  'feature_dir': PosixPath('data/fbank'), 'max_duration': 100, 'bucketing_sampler': False, 'num_buckets': 30, 
  'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 
  'shuffle': True, 'return_cuts': True, 'num_workers': 2}
  2021-09-26 12:44:31,406 INFO [lexicon.py:113] Loading pre-compiled data/lang_bpe_500/Linv.pt
  2021-09-26 12:44:31,464 INFO [decode.py:548] device: cuda:0
  2021-09-26 12:44:36,171 INFO [checkpoint.py:92] Loading checkpoint from conformer_ctc/exp/epoch-25.pt
  2021-09-26 12:44:36,776 INFO [decode.py:652] Number of model parameters: 109226120
  2021-09-26 12:44:37,714 INFO [decode.py:473] batch 0/206, cuts processed until now is 12
  2021-09-26 12:45:15,944 INFO [decode.py:473] batch 100/206, cuts processed until now is 1328
  2021-09-26 12:45:54,443 INFO [decode.py:473] batch 200/206, cuts processed until now is 2563
  2021-09-26 12:45:56,411 INFO [decode.py:494] The transcripts are stored in conformer_ctc/exp/recogs-test-clean-ctc-decoding.txt
  2021-09-26 12:45:56,592 INFO [utils.py:331] [test-clean-ctc-decoding] %WER 3.26% [1715 / 52576, 163 ins, 128 del, 1424 sub ]
  2021-09-26 12:45:56,807 INFO [decode.py:506] Wrote detailed error stats to conformer_ctc/exp/errs-test-clean-ctc-decoding.txt
  2021-09-26 12:45:56,808 INFO [decode.py:522]
  For test-clean, WER of different settings are:
  ctc-decoding    3.26    best for test-clean

  2021-09-26 12:45:57,362 INFO [decode.py:473] batch 0/203, cuts processed until now is 15
  2021-09-26 12:46:35,565 INFO [decode.py:473] batch 100/203, cuts processed until now is 1477
  2021-09-26 12:47:15,106 INFO [decode.py:473] batch 200/203, cuts processed until now is 2922
  2021-09-26 12:47:16,131 INFO [decode.py:494] The transcripts are stored in conformer_ctc/exp/recogs-test-other-ctc-decoding.txt
  2021-09-26 12:47:16,208 INFO [utils.py:331] [test-other-ctc-decoding] %WER 8.21% [4295 / 52343, 396 ins, 315 del, 3584 sub ]
  2021-09-26 12:47:16,432 INFO [decode.py:506] Wrote detailed error stats to conformer_ctc/exp/errs-test-other-ctc-decoding.txt
  2021-09-26 12:47:16,432 INFO [decode.py:522]
  For test-other, WER of different settings are:
  ctc-decoding    8.21    best for test-other

  2021-09-26 12:47:16,433 INFO [decode.py:680] Done! 

Pre-trained Model
-----------------

We have uploaded a pre-trained model to
`<https://huggingface.co/pkufool/icefall_asr_librispeech_conformer_ctc>`_.

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

  $ cd egs/librispeech/ASR
  $ mkdir tmp
  $ cd tmp
  $ git lfs install
  $ git clone https://huggingface.co/pkufool/icefall_asr_librispeech_conformer_ctc

.. CAUTION::

  You have to use ``git lfs`` to download the pre-trained model.

.. CAUTION::

  In order to use this pre-trained model, your k2 version has to be v1.7 or later.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ tree tmp

.. code-block:: bash

  tmp
  `-- icefall_asr_librispeech_conformer_ctc
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
      |   `-- pretrained.pt
      `-- test_wavs
          |-- 1089-134686-0001.flac
          |-- 1221-135766-0001.flac
          |-- 1221-135766-0002.flac
          `-- trans.txt

  6 directories, 11 files

**File descriptions**:

  - ``data/lang_bpe/HLG.pt``

      It is the decoding graph.

  - ``data/lang_bpe/bpe.model``

      It is a sentencepiece model. You can use it to reproduce our results.

  - ``data/lang_bpe/tokens.txt``

      It contains tokens and their IDs, generated from ``bpe.model``.
      Provided only for convenience so that you can look up the SOS/EOS ID easily.

  - ``data/lang_bpe/words.txt``

      It contains words and their IDs.

  - ``data/lm/G_4_gram.pt``

      It is a 4-gram LM, used for n-gram LM rescoring.

  - ``exp/pretrained.pt``

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from ``epoch-15.pt`` to ``epoch-34.pt``.
      Note: We have removed optimizer ``state_dict`` to reduce file size.

  - ``test_waves/*.flac``

      It contains some test sound files from LibriSpeech ``test-clean`` dataset.

  - ``test_waves/trans.txt``

      It contains the reference transcripts for the sound files in ``test_waves/``.

The information of the test sound files is listed below:

.. code-block:: bash

  $ soxi tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/*.flac

  Input File     : 'tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:06.62 = 106000 samples ~ 496.875 CDDA sectors
  File Size      : 116k
  Bit Rate       : 140k
  Sample Encoding: 16-bit FLAC

  Input File     : 'tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:16.71 = 267440 samples ~ 1253.62 CDDA sectors
  File Size      : 343k
  Bit Rate       : 164k
  Sample Encoding: 16-bit FLAC

  Input File     : 'tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.83 = 77200 samples ~ 361.875 CDDA sectors
  File Size      : 105k
  Bit Rate       : 174k
  Sample Encoding: 16-bit FLAC

  Total Duration of 3 files: 00:00:28.16

Usage
~~~~~

.. code-block::

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py --help

displays the help information.

It supports three decoding methods:

  - HLG decoding
  - HLG + n-gram LM rescoring
  - HLG + n-gram LM rescoring + attention decoder rescoring

HLG decoding
^^^^^^^^^^^^

HLG decoding uses the best path of the decoding lattice as the decoding result.

The command to run HLG decoding is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_librispeech_conformer_ctc/exp/pretrained.pt \
    --words-file ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/words.txt \
    --HLG ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/HLG.pt \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac

The output is given below:

.. code-block::

  2021-08-20 11:03:05,712 INFO [pretrained.py:217] device: cuda:0
  2021-08-20 11:03:05,712 INFO [pretrained.py:219] Creating model
  2021-08-20 11:03:11,345 INFO [pretrained.py:238] Loading HLG from ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/HLG.pt
  2021-08-20 11:03:18,442 INFO [pretrained.py:255] Constructing Fbank computer
  2021-08-20 11:03:18,444 INFO [pretrained.py:265] Reading sound files: ['./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac', './tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac', './tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac']
  2021-08-20 11:03:18,507 INFO [pretrained.py:271] Decoding started
  2021-08-20 11:03:18,795 INFO [pretrained.py:300] Use HLG decoding
  2021-08-20 11:03:19,149 INFO [pretrained.py:339]
  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED
  BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-08-20 11:03:19,149 INFO [pretrained.py:341] Decoding Done

HLG decoding + LM rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It uses an n-gram LM to rescore the decoding lattice and the best
path of the rescored lattice is the decoding result.

The command to run HLG decoding + LM rescoring is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_librispeech_conformer_ctc/exp/pretrained.pt \
    --words-file ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/words.txt \
    --HLG ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/HLG.pt \
    --method whole-lattice-rescoring \
    --G ./tmp/icefall_asr_librispeech_conformer_ctc/data/lm/G_4_gram.pt \
    --ngram-lm-scale 0.8 \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac

Its output is:

.. code-block::

  2021-08-20 11:12:17,565 INFO [pretrained.py:217] device: cuda:0
  2021-08-20 11:12:17,565 INFO [pretrained.py:219] Creating model
  2021-08-20 11:12:23,728 INFO [pretrained.py:238] Loading HLG from ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/HLG.pt
  2021-08-20 11:12:30,035 INFO [pretrained.py:246] Loading G from ./tmp/icefall_asr_librispeech_conformer_ctc/data/lm/G_4_gram.pt
  2021-08-20 11:13:10,779 INFO [pretrained.py:255] Constructing Fbank computer
  2021-08-20 11:13:10,787 INFO [pretrained.py:265] Reading sound files: ['./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac', './tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac', './tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac']
  2021-08-20 11:13:10,798 INFO [pretrained.py:271] Decoding started
  2021-08-20 11:13:11,085 INFO [pretrained.py:305] Use HLG decoding + LM rescoring
  2021-08-20 11:13:11,736 INFO [pretrained.py:339]
  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED
  BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-08-20 11:13:11,737 INFO [pretrained.py:341] Decoding Done

HLG decoding + LM rescoring + attention decoder rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It uses an n-gram LM to rescore the decoding lattice, extracts
n paths from the rescored lattice, recores the extracted paths with
an attention decoder. The path with the highest score is the decoding result.

The command to run HLG decoding + LM rescoring + attention decoder rescoring is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py \
    --checkpoint ./tmp/icefall_asr_librispeech_conformer_ctc/exp/pretrained.pt \
    --words-file ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/words.txt \
    --HLG ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/HLG.pt \
    --method attention-decoder \
    --G ./tmp/icefall_asr_librispeech_conformer_ctc/data/lm/G_4_gram.pt \
    --ngram-lm-scale 1.3 \
    --attention-decoder-scale 1.2 \
    --nbest-scale 0.5 \
    --num-paths 100 \
    --sos-id 1 \
    --eos-id 1 \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac \
    ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac

The output is below:

.. code-block::

  2021-08-20 11:19:11,397 INFO [pretrained.py:217] device: cuda:0
  2021-08-20 11:19:11,397 INFO [pretrained.py:219] Creating model
  2021-08-20 11:19:17,354 INFO [pretrained.py:238] Loading HLG from ./tmp/icefall_asr_librispeech_conformer_ctc/data/lang_bpe/HLG.pt
  2021-08-20 11:19:24,615 INFO [pretrained.py:246] Loading G from ./tmp/icefall_asr_librispeech_conformer_ctc/data/lm/G_4_gram.pt
  2021-08-20 11:20:04,576 INFO [pretrained.py:255] Constructing Fbank computer
  2021-08-20 11:20:04,584 INFO [pretrained.py:265] Reading sound files: ['./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac', './tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac', './tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac']
  2021-08-20 11:20:04,595 INFO [pretrained.py:271] Decoding started
  2021-08-20 11:20:04,854 INFO [pretrained.py:313] Use HLG + LM rescoring + attention decoder rescoring
  2021-08-20 11:20:05,805 INFO [pretrained.py:339]
  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1089-134686-0001.flac:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0001.flac:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONOURED
  BOSOM TO CONNECT HER PARENT FOR EVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./tmp/icefall_asr_librispeech_conformer_ctc/test_wavs/1221-135766-0002.flac:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-08-20 11:20:05,805 INFO [pretrained.py:341] Decoding Done

Colab notebook
--------------

We do provide a colab notebook for this recipe showing how to use a pre-trained model.

|librispeech asr conformer ctc colab notebook|

.. |librispeech asr conformer ctc colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1huyupXAcHsUrKaWfI83iMEJ6J0Nh0213?usp=sharing

.. HINT::

  Due to limited memory provided by Colab, you have to upgrade to Colab Pro to
  run ``HLG decoding + LM rescoring`` and
  ``HLG decoding + LM rescoring + attention decoder rescoring``.
  Otherwise, you can only run ``HLG decoding`` with Colab.

**Congratulations!** You have finished the librispeech ASR recipe with
conformer CTC models in ``icefall``.
