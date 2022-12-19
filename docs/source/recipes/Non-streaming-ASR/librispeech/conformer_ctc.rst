Conformer CTC
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
  - (5) How to deploy your trained model in C++, without Python dependencies

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

We provide the following YouTube video showing how to run ``./prepare.sh``.

.. note::

   To get the latest news of `next-gen Kaldi <https://github.com/k2-fsa>`_, please subscribe
   the following YouTube channel by `Nadira Povey <https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_:

      `<https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_

..  youtube:: ofEIoJL-mGM


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

    This specifies the decoding method. This script supports 7 decoding methods.
    As for ctc decoding, it uses a sentence piece model to convert word pieces to words.
    And it needs neither a lexicon nor an n-gram LM.

    For example, the following command uses CTC topology for decoding:

    .. code-block::

      $ cd egs/librispeech/ASR
      $ ./conformer_ctc/decode.py --method ctc-decoding --max-duration 300
      # Caution: The above command is tested with a model with vocab size 500.

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

Here are some results for CTC decoding with a vocab size of 500:

Usage:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  # NOTE: Tested with a model with vocab size 500.
  # It won't work for a model with vocab size 5000.
  $ ./conformer_ctc/decode.py \
      --epoch 25 \
      --avg 1 \
      --max-duration 300 \
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
`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09>`_

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

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
  $ cd icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
  $ git lfs pull

.. CAUTION::

  You have to use ``git lfs pull`` to download the pre-trained model.
  Otherwise, you will have the following issue when running ``decode.py``:

    .. code-block::

       _pickle.UnpicklingError: invalid load key, 'v'

  To fix that issue, please use:

     .. code-block:: bash

        cd icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
        git lfs pull

.. CAUTION::

  In order to use this pre-trained model, your k2 version has to be v1.9 or later.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ tree icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09

.. code-block:: bash

  icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
  |-- README.md
  |-- data
  |   |-- lang_bpe_500
  |   |   |-- HLG.pt
  |   |   |-- HLG_modified.pt
  |   |   |-- bpe.model
  |   |   |-- tokens.txt
  |   |   `-- words.txt
  |   `-- lm
  |       `-- G_4_gram.pt
  |-- exp
  |   |-- cpu_jit.pt
  |   `-- pretrained.pt
  |-- log
  |   `-- log-decode-2021-11-09-17-38-28
  `-- test_wavs
      |-- 1089-134686-0001.wav
      |-- 1221-135766-0001.wav
      |-- 1221-135766-0002.wav
      `-- trans.txt


**File descriptions**:
  - ``data/lang_bpe_500/HLG.pt``

      It is the decoding graph.

  - ``data/lang_bpe_500/HLG_modified.pt``

      It uses a modified CTC topology while building HLG.

  - ``data/lang_bpe_500/bpe.model``

      It is a sentencepiece model. You can use it to reproduce our results.

  - ``data/lang_bpe_500/tokens.txt``

      It contains tokens and their IDs, generated from ``bpe.model``.
      Provided only for convenience so that you can look up the SOS/EOS ID easily.

  - ``data/lang_bpe_500/words.txt``

      It contains words and their IDs.

  - ``data/lm/G_4_gram.pt``

      It is a 4-gram LM, used for n-gram LM rescoring.

  - ``exp/pretrained.pt``

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from ``epoch-23.pt`` to ``epoch-77.pt``.
      Note: We have removed optimizer ``state_dict`` to reduce file size.

  - ``exp/cpu_jit.pt``

      It contains torch scripted model that can be deployed in C++.

  - ``test_wavs/*.wav``

      It contains some test sound files from LibriSpeech ``test-clean`` dataset.

  - ``test_wavs/trans.txt``

      It contains the reference transcripts for the sound files in ``test_wavs/``.

The information of the test sound files is listed below:

.. code-block:: bash

  $ soxi icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/*.wav

  Input File     : 'icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:06.62 = 106000 samples ~ 496.875 CDDA sectors
  File Size      : 212k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:16.71 = 267440 samples ~ 1253.62 CDDA sectors
  File Size      : 535k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.83 = 77200 samples ~ 361.875 CDDA sectors
  File Size      : 154k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM

  Total Duration of 3 files: 00:00:28.16

Usage
~~~~~

.. code-block::

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py --help

displays the help information.

It supports 4 decoding methods:

  - CTC decoding
  - HLG decoding
  - HLG + n-gram LM rescoring
  - HLG + n-gram LM rescoring + attention decoder rescoring

CTC decoding
^^^^^^^^^^^^

CTC decoding uses the best path of the decoding lattice as the decoding result
without any LM or lexicon.

The command to run CTC decoding is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py \
     --checkpoint ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt \
     --bpe-model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/bpe.model \
     --method ctc-decoding \
     --num-classes 500 \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

The output is given below:

.. code-block::

  2021-11-10 12:12:29,554 INFO [pretrained.py:260] {'sample_rate': 16000, 'subsampling_factor': 4, 'vgg_frontend': False, 'use_feat_batchnorm': True, 'feature_dim': 80, 'nhead': 8, 'attention_dim': 512, 'num_decoder_layers': 0, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt', 'words_file': None, 'HLG': None, 'bpe_model': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/bpe.model', 'method': 'ctc-decoding', 'G': None, 'num_paths': 100, 'ngram_lm_scale': 1.3, 'attention_decoder_scale': 1.2, 'nbest_scale': 0.5, 'sos_id': 1, 'num_classes': 500, 'eos_id': 1, 'sound_files': ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav'], 'env_info': {'k2-version': '1.9', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '7178d67e594bc7fa89c2b331ad7bd1c62a6a9eb4', 'k2-git-date': 'Tue Oct 26 22:12:54 2021', 'lhotse-version': '0.11.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.1', 'python-version': '3.8', 'icefall-git-branch': 'bpe-500', 'icefall-git-sha1': '8d93169-dirty', 'icefall-git-date': 'Wed Nov 10 11:52:44 2021', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-fix', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-bpe-500/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-bpe-500/lhotse/__init__.py'}}
  2021-11-10 12:12:29,554 INFO [pretrained.py:266] device: cuda:0
  2021-11-10 12:12:29,554 INFO [pretrained.py:268] Creating model
  2021-11-10 12:12:35,600 INFO [pretrained.py:285] Constructing Fbank computer
  2021-11-10 12:12:35,601 INFO [pretrained.py:295] Reading sound files: ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav']
  2021-11-10 12:12:35,758 INFO [pretrained.py:301] Decoding started
  2021-11-10 12:12:36,025 INFO [pretrained.py:319] Use CTC decoding
  2021-11-10 12:12:36,204 INFO [pretrained.py:425]
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROFFELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED B
  OSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-11-10 12:12:36,204 INFO [pretrained.py:427] Decoding Done

HLG decoding
^^^^^^^^^^^^

HLG decoding uses the best path of the decoding lattice as the decoding result.

The command to run HLG decoding is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py \
     --checkpoint ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt \
     --words-file ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt \
     --method 1best \
     --num-classes 500 \
     --HLG ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

The output is given below:

.. code-block::

  2021-11-10 13:33:03,723 INFO [pretrained.py:260] {'sample_rate': 16000, 'subsampling_factor': 4, 'vgg_frontend': False, 'use_feat_batchnorm': True, 'feature_dim': 80, 'nhead': 8, 'attention_dim': 512, 'num_decoder_layers': 0, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt', 'words_file': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt', 'HLG': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt', 'bpe_model': None, 'method': '1best', 'G': None, 'num_paths': 100, 'ngram_lm_scale': 1.3, 'attention_decoder_scale': 1.2, 'nbest_scale': 0.5, 'sos_id': 1, 'num_classes': 500, 'eos_id': 1, 'sound_files': ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav'], 'env_info': {'k2-version': '1.9', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '7178d67e594bc7fa89c2b331ad7bd1c62a6a9eb4', 'k2-git-date': 'Tue Oct 26 22:12:54 2021', 'lhotse-version': '0.11.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.1', 'python-version': '3.8', 'icefall-git-branch': 'bpe-500', 'icefall-git-sha1': '8d93169-dirty', 'icefall-git-date': 'Wed Nov 10 11:52:44 2021', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-fix', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-bpe-500/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-bpe-500/lhotse/__init__.py'}}
  2021-11-10 13:33:03,723 INFO [pretrained.py:266] device: cuda:0
  2021-11-10 13:33:03,723 INFO [pretrained.py:268] Creating model
  2021-11-10 13:33:09,775 INFO [pretrained.py:285] Constructing Fbank computer
  2021-11-10 13:33:09,776 INFO [pretrained.py:295] Reading sound files: ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav']
  2021-11-10 13:33:09,881 INFO [pretrained.py:301] Decoding started
  2021-11-10 13:33:09,951 INFO [pretrained.py:352] Loading HLG from ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt
  2021-11-10 13:33:13,234 INFO [pretrained.py:384] Use HLG decoding
  2021-11-10 13:33:13,571 INFO [pretrained.py:425]
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-11-10 13:33:13,571 INFO [pretrained.py:427] Decoding Done


HLG decoding + LM rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It uses an n-gram LM to rescore the decoding lattice and the best
path of the rescored lattice is the decoding result.

The command to run HLG decoding + LM rescoring is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  ./conformer_ctc/pretrained.py \
     --checkpoint ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt \
     --words-file ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt \
     --method whole-lattice-rescoring \
     --num-classes 500 \
     --HLG ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
     --G ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt \
     --ngram-lm-scale 1.0 \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

Its output is:

.. code-block::

  2021-11-10 13:39:55,857 INFO [pretrained.py:260] {'sample_rate': 16000, 'subsampling_factor': 4, 'vgg_frontend': False, 'use_feat_batchnorm': True, 'feature_dim': 80, 'nhead': 8, 'attention_dim': 512, 'num_decoder_layers': 0, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt', 'words_file': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt', 'HLG': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt', 'bpe_model': None, 'method': 'whole-lattice-rescoring', 'G': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt', 'num_paths': 100, 'ngram_lm_scale': 1.0, 'attention_decoder_scale': 1.2, 'nbest_scale': 0.5, 'sos_id': 1, 'num_classes': 500, 'eos_id': 1, 'sound_files': ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav'], 'env_info': {'k2-version': '1.9', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-$it-sha1': '7178d67e594bc7fa89c2b331ad7bd1c62a6a9eb4', 'k2-git-date': 'Tue Oct 26 22:12:54 2021', 'lhotse-version': '0.11.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.1', 'python-version': '3.8', 'icefall-git-branch': 'bpe-500', 'icefall-git-sha1': '8d93169-dirty', 'icefall-git-date': 'Wed Nov 10 11:52:44 2021', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-fix', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-bpe-500/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-bpe-500/lhotse/__init__.py'}}
  2021-11-10 13:39:55,858 INFO [pretrained.py:266] device: cuda:0
  2021-11-10 13:39:55,858 INFO [pretrained.py:268] Creating model
  2021-11-10 13:40:01,979 INFO [pretrained.py:285] Constructing Fbank computer
  2021-11-10 13:40:01,980 INFO [pretrained.py:295] Reading sound files: ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav']
  2021-11-10 13:40:02,055 INFO [pretrained.py:301] Decoding started
  2021-11-10 13:40:02,117 INFO [pretrained.py:352] Loading HLG from ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt
  2021-11-10 13:40:05,051 INFO [pretrained.py:363] Loading G from ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt
  2021-11-10 13:40:18,959 INFO [pretrained.py:389] Use HLG decoding + LM rescoring
  2021-11-10 13:40:19,546 INFO [pretrained.py:425]
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-11-10 13:40:19,546 INFO [pretrained.py:427] Decoding Done


HLG decoding + LM rescoring + attention decoder rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It uses an n-gram LM to rescore the decoding lattice, extracts
n paths from the rescored lattice, recores the extracted paths with
an attention decoder. The path with the highest score is the decoding result.

The command to run HLG decoding + LM rescoring + attention decoder rescoring is:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./conformer_ctc/pretrained.py \
     --checkpoint ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt \
     --words-file ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt \
     --method attention-decoder \
     --num-classes 500 \
     --HLG ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
     --G ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt \
     --ngram-lm-scale 2.0 \
     --attention-decoder-scale 2.0 \
     --nbest-scale 0.5 \
     --num-paths 100 \
     --sos-id 1 \
     --eos-id 1 \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
     ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

The output is below:

.. code-block::

  2021-11-10 13:43:45,598 INFO [pretrained.py:260] {'sample_rate': 16000, 'subsampling_factor': 4, 'vgg_frontend': False, 'use_feat_batchnorm': True, 'feature_dim': 80, 'nhead': 8, 'attention_dim': 512, 'num_decoder_layers': 6, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/pretrained.pt', 'words_file': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt', 'HLG': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt', 'bpe_model': None, 'method': 'attention-decoder', 'G': './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt', 'num_paths': 100, 'ngram_lm_scale': 2.0, 'attention_decoder_scale': 2.0, 'nbest_scale': 0.5, 'sos_id': 1, 'num_classes': 500, 'eos_id': 1, 'sound_files': ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav'], 'env_info': {'k2-version': '1.9', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '7178d67e594bc7fa89c2b331ad7bd1c62a6a9eb4', 'k2-git-date': 'Tue Oct 26 22:12:54 2021', 'lhotse-version': '0.11.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.1', 'python-version': '3.8', 'icefall-git-branch': 'bpe-500', 'icefall-git-sha1': '8d93169-dirty', 'icefall-git-date': 'Wed Nov 10 11:52:44 2021', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-fix', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-bpe-500/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-bpe-500/lhotse/__init__.py'}}
  2021-11-10 13:43:45,599 INFO [pretrained.py:266] device: cuda:0
  2021-11-10 13:43:45,599 INFO [pretrained.py:268] Creating model
  2021-11-10 13:43:51,833 INFO [pretrained.py:285] Constructing Fbank computer
  2021-11-10 13:43:51,834 INFO [pretrained.py:295] Reading sound files: ['./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav', './icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav']
  2021-11-10 13:43:51,915 INFO [pretrained.py:301] Decoding started
  2021-11-10 13:43:52,076 INFO [pretrained.py:352] Loading HLG from ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt
  2021-11-10 13:43:55,110 INFO [pretrained.py:363] Loading G from ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt
  2021-11-10 13:44:09,329 INFO [pretrained.py:397] Use HLG + LM rescoring + attention decoder rescoring
  2021-11-10 13:44:10,192 INFO [pretrained.py:425]
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav:
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav:
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav:
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

  2021-11-10 13:44:10,192 INFO [pretrained.py:427] Decoding Done


Compute WER with the pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To check the WER of the pre-trained model on the test datasets, run:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ cd icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/
  $ ln -s pretrained.pt epoch-999.pt
  $ cd ../..
  $ ./conformer_ctc/decode.py \
      --exp-dir ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp \
      --lang-dir ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500 \
      --lm-dir ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm \
      --epoch 999 \
      --avg 1 \
      --concatenate-cuts 0 \
      --bucketing-sampler 1 \
      --max-duration 30 \
      --num-paths 1000 \
      --method attention-decoder \
      --nbest-scale 0.5


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

**Congratulations!** You have finished the LibriSpeech ASR recipe with
conformer CTC models in ``icefall``.

If you want to deploy your trained model in C++, please read the following section.

Deployment with C++
-------------------

This section describes how to deploy the pre-trained model in C++, without
Python dependencies.

.. HINT::

  At present, it does NOT support streaming decoding.

First, let us compile k2 from source:

.. code-block:: bash

  $ cd $HOME
  $ git clone https://github.com/k2-fsa/k2
  $ cd k2
  $ git checkout v2.0-pre

.. CAUTION::

  You have to switch to the branch ``v2.0-pre``!

.. code-block:: bash

  $ mkdir build-release
  $ cd build-release
  $ cmake -DCMAKE_BUILD_TYPE=Release ..
  $ make -j ctc_decode hlg_decode ngram_lm_rescore attention_rescore

  # You will find four binaries in `./bin`, i.e.,
  # ./bin/ctc_decode, ./bin/hlg_decode,
  # ./bin/ngram_lm_rescore, and ./bin/attention_rescore

Now you are ready to go!

Assume you have run:

  .. code-block:: bash

    $ cd k2/build-release
    $ ln -s /path/to/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09 ./

To view the usage of ``./bin/ctc_decode``, run:

.. code-block::

  $ ./bin/ctc_decode

It will show you the following message:

.. code-block:: bash

  Please provide --nn_model

  This file implements decoding with a CTC topology, without any
  kinds of LM or lexicons.

  Usage:
    ./bin/ctc_decode \
      --use_gpu true \
      --nn_model <path to torch scripted pt file> \
      --bpe_model <path to pre-trained BPE model> \
      <path to foo.wav> \
      <path to bar.wav> \
      <more waves if any>

  To see all possible options, use
    ./bin/ctc_decode --help

  Caution:
   - Only sound files (*.wav) with single channel are supported.
   - It assumes the model is conformer_ctc/transformer.py from icefall.
     If you use a different model, you have to change the code
     related to `model.forward` in this file.


CTC decoding
^^^^^^^^^^^^

.. code-block:: bash

  ./bin/ctc_decode \
    --use_gpu true \
    --nn_model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
    --bpe_model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/bpe.model \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

Its output is:

.. code-block::

  2021-11-10 13:57:55.316 [I] k2/torch/bin/ctc_decode.cu:105:int main(int, char**) Use GPU
  2021-11-10 13:57:55.316 [I] k2/torch/bin/ctc_decode.cu:109:int main(int, char**) Device: cuda:0
  2021-11-10 13:57:55.316 [I] k2/torch/bin/ctc_decode.cu:118:int main(int, char**) Load wave files
  2021-11-10 13:58:01.221 [I] k2/torch/bin/ctc_decode.cu:125:int main(int, char**) Build Fbank computer
  2021-11-10 13:58:01.222 [I] k2/torch/bin/ctc_decode.cu:136:int main(int, char**) Compute features
  2021-11-10 13:58:01.228 [I] k2/torch/bin/ctc_decode.cu:144:int main(int, char**) Load neural network model
  2021-11-10 13:58:02.19 [I] k2/torch/bin/ctc_decode.cu:159:int main(int, char**) Compute nnet_output
  2021-11-10 13:58:02.543 [I] k2/torch/bin/ctc_decode.cu:174:int main(int, char**) Build CTC topo
  2021-11-10 13:58:02.547 [I] k2/torch/bin/ctc_decode.cu:177:int main(int, char**) Decoding
  2021-11-10 13:58:02.708 [I] k2/torch/bin/ctc_decode.cu:207:int main(int, char**)
  Decoding result:

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROFFELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

HLG decoding
^^^^^^^^^^^^

.. code-block:: bash

  ./bin/hlg_decode \
    --use_gpu true \
    --nn_model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
    --hlg ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
    --word_table ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

The output is:

.. code-block::

  2021-11-10 13:59:04.729 [I] k2/torch/bin/hlg_decode.cu:111:int main(int, char**) Use GPU
  2021-11-10 13:59:04.729 [I] k2/torch/bin/hlg_decode.cu:115:int main(int, char**) Device: cuda:0
  2021-11-10 13:59:04.729 [I] k2/torch/bin/hlg_decode.cu:124:int main(int, char**) Load wave files
  2021-11-10 13:59:10.702 [I] k2/torch/bin/hlg_decode.cu:131:int main(int, char**) Build Fbank computer
  2021-11-10 13:59:10.703 [I] k2/torch/bin/hlg_decode.cu:142:int main(int, char**) Compute features
  2021-11-10 13:59:10.707 [I] k2/torch/bin/hlg_decode.cu:150:int main(int, char**) Load neural network model
  2021-11-10 13:59:11.545 [I] k2/torch/bin/hlg_decode.cu:165:int main(int, char**) Compute nnet_output
  2021-11-10 13:59:12.72 [I] k2/torch/bin/hlg_decode.cu:180:int main(int, char**) Load ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt
  2021-11-10 13:59:12.994 [I] k2/torch/bin/hlg_decode.cu:185:int main(int, char**) Decoding
  2021-11-10 13:59:13.268 [I] k2/torch/bin/hlg_decode.cu:216:int main(int, char**)
  Decoding result:

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


HLG decoding + n-gram LM rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  ./bin/ngram_lm_rescore \
    --use_gpu true \
    --nn_model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
    --hlg ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
    --g ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt \
    --ngram_lm_scale 1.0 \
    --word_table ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

The output is:

.. code-block::

  2021-11-10 14:00:55.279 [I] k2/torch/bin/ngram_lm_rescore.cu:122:int main(int, char**) Use GPU
  2021-11-10 14:00:55.280 [I] k2/torch/bin/ngram_lm_rescore.cu:126:int main(int, char**) Device: cuda:0
  2021-11-10 14:00:55.280 [I] k2/torch/bin/ngram_lm_rescore.cu:135:int main(int, char**) Load wave files
  2021-11-10 14:01:01.214 [I] k2/torch/bin/ngram_lm_rescore.cu:142:int main(int, char**) Build Fbank computer
  2021-11-10 14:01:01.215 [I] k2/torch/bin/ngram_lm_rescore.cu:153:int main(int, char**) Compute features
  2021-11-10 14:01:01.219 [I] k2/torch/bin/ngram_lm_rescore.cu:161:int main(int, char**) Load neural network model
  2021-11-10 14:01:01.945 [I] k2/torch/bin/ngram_lm_rescore.cu:176:int main(int, char**) Compute nnet_output
  2021-11-10 14:01:02.475 [I] k2/torch/bin/ngram_lm_rescore.cu:191:int main(int, char**) Load ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt
  2021-11-10 14:01:03.398 [I] k2/torch/bin/ngram_lm_rescore.cu:199:int main(int, char**) Decoding
  2021-11-10 14:01:03.515 [I] k2/torch/bin/ngram_lm_rescore.cu:205:int main(int, char**) Load n-gram LM: ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt
  2021-11-10 14:01:07.432 [W] k2/torch/csrc/deserialization.cu:441:k2::FsaClass k2::LoadFsa(const string&, c10::optional<c10::Device>)
  Ignore non tensor attribute: 'dummy' of type: Int
  2021-11-10 14:01:07.589 [I] k2/torch/bin/ngram_lm_rescore.cu:214:int main(int, char**) Rescore with an n-gram LM
  2021-11-10 14:01:08.68 [I] k2/torch/bin/ngram_lm_rescore.cu:242:int main(int, char**)
  Decoding result:

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION


HLG decoding + n-gram LM rescoring + attention decoder rescoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  ./bin/attention_rescore \
    --use_gpu true \
    --nn_model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
    --hlg ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
    --g ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt \
    --ngram_lm_scale 2.0 \
    --attention_scale 2.0 \
    --num_paths 100 \
    --nbest_scale 0.5 \
    --word_table ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/words.txt \
    --sos_id 1 \
    --eos_id 1 \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
    ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

The output is:

.. code-block::

  2021-11-10 14:02:43.656 [I] k2/torch/bin/attention_rescore.cu:149:int main(int, char**) Use GPU
  2021-11-10 14:02:43.656 [I] k2/torch/bin/attention_rescore.cu:153:int main(int, char**) Device: cuda:0
  2021-11-10 14:02:43.656 [I] k2/torch/bin/attention_rescore.cu:162:int main(int, char**) Load wave files
  2021-11-10 14:02:49.216 [I] k2/torch/bin/attention_rescore.cu:169:int main(int, char**) Build Fbank computer
  2021-11-10 14:02:49.217 [I] k2/torch/bin/attention_rescore.cu:180:int main(int, char**) Compute features
  2021-11-10 14:02:49.222 [I] k2/torch/bin/attention_rescore.cu:188:int main(int, char**) Load neural network model
  2021-11-10 14:02:49.984 [I] k2/torch/bin/attention_rescore.cu:203:int main(int, char**) Compute nnet_output
  2021-11-10 14:02:50.624 [I] k2/torch/bin/attention_rescore.cu:220:int main(int, char**) Load ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt
  2021-11-10 14:02:51.519 [I] k2/torch/bin/attention_rescore.cu:228:int main(int, char**) Decoding
  2021-11-10 14:02:51.632 [I] k2/torch/bin/attention_rescore.cu:234:int main(int, char**) Load n-gram LM: ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lm/G_4_gram.pt
  2021-11-10 14:02:55.537 [W] k2/torch/csrc/deserialization.cu:441:k2::FsaClass k2::LoadFsa(const string&, c10::optional<c10::Device>) Ignore non tensor attribute: 'dummy' of type: Int
  2021-11-10 14:02:55.645 [I] k2/torch/bin/attention_rescore.cu:243:int main(int, char**) Rescore with an n-gram LM
  2021-11-10 14:02:55.970 [I] k2/torch/bin/attention_rescore.cu:246:int main(int, char**) Sample 100 paths
  2021-11-10 14:02:56.215 [I] k2/torch/bin/attention_rescore.cu:293:int main(int, char**) Run attention decoder
  2021-11-10 14:02:57.35 [I] k2/torch/bin/attention_rescore.cu:303:int main(int, char**) Rescoring
  2021-11-10 14:02:57.179 [I] k2/torch/bin/attention_rescore.cu:369:int main(int, char**)
  Decoding result:

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav
  AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav
  GOD AS A DIRECT CONSEQUENCE OF THE SIN WHICH MAN THUS PUNISHED HAD GIVEN HER A LOVELY CHILD WHOSE PLACE WAS ON THAT SAME DISHONORED BOSOM TO CONNECT HER PARENT FOREVER WITH THE RACE AND DESCENT OF MORTALS AND TO BE FINALLY A BLESSED SOUL IN HEAVEN

  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav
  YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION

There is a Colab notebook showing you how to run a torch scripted model in C++.
Please see |librispeech asr conformer ctc torch script colab notebook|

.. |librispeech asr conformer ctc torch script colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1BIGLWzS36isskMXHKcqC9ysN6pspYXs_?usp=sharing
