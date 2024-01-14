Stateless Transducer
====================

This tutorial shows you how to do transducer training in ``icefall``.

.. HINT::

  Instead of using RNN-T or RNN transducer, we only use transducer
  here. As you will see, there are no RNNs in the model.

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe.

In this tutorial, you will learn:

  - (1) What does the transducer model look like
  - (2) How to prepare data for training and decoding
  - (3) How to start the training, either with a single GPU or with multiple GPUs
  - (4) How to do decoding after training, with greedy search, beam search and, **modified beam search**
  - (5) How to use a pre-trained model provided by us to transcribe sound files


The Model
---------

The transducer model consists of 3 parts:

- **Encoder**: It is a conformer encoder with the following parameters

    - Number of heads: 8
    - Attention dim: 512
    - Number of layers: 12
    - Feedforward dim: 2048

- **Decoder**: We use a stateless model consisting of:

    - An embedding layer with embedding dim 512
    - A Conv1d layer with a default kernel size 2 (i.e. it sees 2
      symbols of left-context by default)

- **Joiner**: It consists of a ``nn.tanh()`` and a ``nn.Linear()``.

.. Caution::

  The decoder is stateless and very simple. It is borrowed from
  `<https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419>`_
  (Rnn-Transducer with Stateless Prediction Network)

  We make one modification to it: Place a Conv1d layer right after
  the embedding layer.

When using Chinese characters as modelling unit, whose vocabulary size
is 4336 in this specific dataset,
the number of parameters of the model is ``87939824``, i.e., about ``88 M``.

The Loss
--------

We are using `<https://github.com/csukuangfj/optimized_transducer>`_
to compute the transducer loss, which removes extra paddings
in loss computation to save memory.

.. Hint::

  ``optimized_transducer`` implements the technqiues proposed
  in `Improving RNN Transducer Modeling for End-to-End Speech Recognition <https://arxiv.org/abs/1909.12415>`_ to save memory.

  Furthermore, it supports ``modified transducer``, limiting the maximum
  number of symbols that can be emitted per frame to 1, which simplifies
  the decoding process significantly. Also, the experiment results
  show that it does not degrade the performance.

  See `<https://github.com/csukuangfj/optimized_transducer#modified-transducer>`_
  for what exactly modified transducer is.

  `<https://github.com/csukuangfj/transducer-loss-benchmarking>`_ shows that
  in the unpruned case ``optimized_transducer`` has the advantage about minimizing
  memory usage.

.. todo::

  Add tutorial about ``pruned_transducer_stateless`` that uses k2
  pruned transducer loss.

.. hint::

  You can use::

    pip install optimized_transducer

  to install ``optimized_transducer``. Refer to
  `<https://github.com/csukuangfj/optimized_transducer>`_ for other
  alternatives.

Data Preparation
----------------

To prepare the data for training, please use the following commands:

.. code-block:: bash

  cd egs/aishell/ASR
  ./prepare.sh --stop-stage 4
  ./prepare.sh --stage 6 --stop-stage 6

.. note::

  You can use ``./prepare.sh``, though it will generate FSTs that
  are not used in transducer training.

When you finish running the script, you will get the following two folders:

  - ``data/fbank``: It saves the pre-computed features
  - ``data/lang_char``: It contains tokens that will be used in the training

Training
--------

.. code-block:: bash

  cd egs/aishell/ASR
  ./transducer_stateless_modified/train.py --help

shows you the training options that can be passed from the commandline.
The following options are used quite often:

  - ``--exp-dir``

    The experiment folder to save logs and model checkpoints,
    defaults to ``./transducer_stateless_modified/exp``.

  - ``--num-epochs``

    It is the number of epochs to train. For instance,
    ``./transducer_stateless_modified/train.py --num-epochs 30`` trains for 30
    epochs and generates ``epoch-0.pt``, ``epoch-1.pt``, ..., ``epoch-29.pt``
    in the folder set by ``--exp-dir``.

  - ``--start-epoch``

    It's used to resume training.
    ``./transducer_stateless_modified/train.py --start-epoch 10`` loads the
    checkpoint from ``exp_dir/epoch-9.pt`` and starts
    training from epoch 10, based on the state from epoch 9.

  - ``--world-size``

    It is used for single-machine multi-GPU DDP training.

      - (a) If it is 1, then no DDP training is used.

      - (b) If it is 2, then GPU 0 and GPU 1 are used for DDP training.

    The following shows some use cases with it.

      **Use case 1**: You have 4 GPUs, but you only want to use GPU 0 and
      GPU 2 for training. You can do the following:

        .. code-block:: bash

          $ cd egs/aishell/ASR
          $ export CUDA_VISIBLE_DEVICES="0,2"
          $ ./transducer_stateless_modified/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/aishell/ASR
          $ ./transducer_stateless_modified/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/aishell/ASR
          $ export CUDA_VISIBLE_DEVICES="3"
          $ ./transducer_stateless_modified/train.py --world-size 1

    .. CAUTION::

      Only single-machine multi-GPU DDP training is implemented at present.
      There is an on-going PR `<https://github.com/k2-fsa/icefall/pull/63>`_
      that adds support for multi-machine multi-GPU DDP training.

  - ``--max-duration``

    It specifies the number of seconds over all utterances in a
    batch **before padding**.
    If you encounter CUDA OOM, please reduce it. For instance, if
    your are using V100 NVIDIA GPU with 32 GB RAM, we recommend you
    to set it to ``300`` when the vocabulary size is 500.

    .. HINT::

      Due to padding, the number of seconds of all utterances in a
      batch will usually be larger than ``--max-duration``.

      A larger value for ``--max-duration`` may cause OOM during training,
      while a smaller value may increase the training time. You have to
      tune it.

  - ``--lr-factor``

    It controls the learning rate. If you use a single GPU for training, you
    may want to use a small value for it. If you use multiple GPUs for training,
    you may increase it.

  - ``--context-size``

    It specifies the kernel size in the decoder. The default value 2 means it
    functions as a tri-gram LM.

  - ``--modified-transducer-prob``

    It specifies the probability to use modified transducer loss.
    If it is 0, then no modified transducer is used; if it is 1,
    then it uses modified transducer loss for all batches. If it is
    ``p``, it applies modified transducer with probability ``p``.

There are some training options, e.g.,
number of warmup steps,
that are not passed from the commandline.
They are pre-configured by the function ``get_params()`` in
`transducer_stateless_modified/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/transducer_stateless_modified/train.py#L162>`_

If you need to change them, please modify ``./transducer_stateless_modified/train.py`` directly.

.. CAUTION::

  The training set is perturbed by speed with two factors: 0.9 and 1.1.
  Each epoch actually processes ``3x150 == 450`` hours of data.

Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in the folder set by ``--exp-dir``
(defaults to ``transducer_stateless_modified/exp``). You will find the following files in that directory:

  - ``epoch-0.pt``, ``epoch-1.pt``, ...

    These are checkpoint files, containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./transducer_stateless_modified/train.py --start-epoch 11

  - ``tensorboard/``

    This folder contains TensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd transducer_stateless_modified/exp/tensorboard
        $ tensorboard dev upload --logdir . --name "Aishell transducer training with icefall" --description "Training modified transducer, see https://github.com/k2-fsa/icefall/pull/219"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/laGZ6HrcQxOigbFD5E0Y3Q/

        [2022-03-03T14:29:45] Started scanning logdir.
        [2022-03-03T14:29:48] Total uploaded: 8477 scalars, 0 tensors, 0 binary objects
        Listening for new data in logdir...

    Note there is a `URL <https://tensorboard.dev/experiment/laGZ6HrcQxOigbFD5E0Y3Q/>`_ in the
    above output, click it and you will see the following screenshot:

      .. figure:: images/aishell-transducer_stateless_modified-tensorboard-log.png
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/laGZ6HrcQxOigbFD5E0Y3Q

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
  $ ./transducer_stateless_modified/train.py --max-duration 250

It uses ``--max-duration`` of 250 to avoid OOM.


**Case 2**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ export CUDA_VISIBLE_DEVICES="0,3"
  $ ./transducer_stateless_modified/train.py --world-size 2

It uses GPU 0 and GPU 3 for DDP training.

**Case 3**
^^^^^^^^^^

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./transducer_stateless_modified/train.py --num-epochs 10 --start-epoch 3

It loads checkpoint ``./transducer_stateless_modified/exp/epoch-2.pt`` and starts
training from epoch 3. Also, it trains for 10 epochs.

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ ./transducer_stateless_modified/decode.py --help

shows the options for decoding.

The commonly used options are:

  - ``--method``

    This specifies the decoding method. Currently, it supports:

      - **greedy_search**. You can provide the commandline option ``--max-sym-per-frame``
        to limit the maximum number of symbols that can be emitted per frame.

      - **beam_search**. You can provide the commandline option ``--beam-size``.

      - **modified_beam_search**. You can also provide the commandline option ``--beam-size``.
        To use this method, we assume that you have trained your model with modified transducer,
        i.e., used the option ``--modified-transducer-prob`` in the training.

    The following command uses greedy search for decoding

    .. code-block::

      $ cd egs/aishell/ASR
      $ ./transducer_stateless_modified/decode.py \
              --epoch 64 \
              --avg 33 \
              --exp-dir ./transducer_stateless_modified/exp \
              --max-duration 100 \
              --decoding-method greedy_search \
              --max-sym-per-frame 1

    The following command uses beam search for decoding

    .. code-block::

      $ cd egs/aishell/ASR
      $ ./transducer_stateless_modified/decode.py \
              --epoch 64 \
              --avg 33 \
              --exp-dir ./transducer_stateless_modified/exp \
              --max-duration 100 \
              --decoding-method beam_search \
              --beam-size 4

    The following command uses ``modified`` beam search for decoding

    .. code-block::

      $ cd egs/aishell/ASR
      $ ./transducer_stateless_modified/decode.py \
              --epoch 64 \
              --avg 33 \
              --exp-dir ./transducer_stateless_modified/exp \
              --max-duration 100 \
              --decoding-method modified_beam_search \
              --beam-size 4

  - ``--max-duration``

    It has the same meaning as the one used in training. A larger
    value may cause OOM.

  - ``--epoch``

    It specifies the checkpoint from which epoch that should be used for decoding.

  - ``--avg``

    It specifies the number of models to average. For instance, if it is 3 and if
    ``--epoch=10``, then it averages the checkpoints ``epoch-8.pt``, ``epoch-9.pt``,
    and ``epoch-10.pt`` and the averaged checkpoint is used for decoding.

After decoding, you can find the decoding logs and results in `exp_dir/log/<decoding_method>`, e.g.,
``exp_dir/log/greedy_search``.

Pre-trained Model
-----------------

We have uploaded a pre-trained model to
`<https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2022-03-01>`_

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
  $ git clone https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2022-03-01


.. CAUTION::

  You have to use ``git lfs`` to download the pre-trained model.

After downloading, you will have the following files:

.. code-block:: bash

  $ cd egs/aishell/ASR
  $ tree tmp/icefall-aishell-transducer-stateless-modified-2022-03-01


.. code-block:: bash

  tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/
  |-- README.md
  |-- data
  |   `-- lang_char
  |       |-- L.pt
  |       |-- lexicon.txt
  |       |-- tokens.txt
  |       `-- words.txt
  |-- exp
  |   `-- pretrained.pt
  |-- log
  |   |-- errs-test-beam_4-epoch-64-avg-33-beam-4.txt
  |   |-- errs-test-greedy_search-epoch-64-avg-33-context-2-max-sym-per-frame-1.txt
  |   |-- log-decode-epoch-64-avg-33-beam-4-2022-03-02-12-05-03
  |   |-- log-decode-epoch-64-avg-33-context-2-max-sym-per-frame-1-2022-02-28-18-13-07
  |   |-- recogs-test-beam_4-epoch-64-avg-33-beam-4.txt
  |   `-- recogs-test-greedy_search-epoch-64-avg-33-context-2-max-sym-per-frame-1.txt
  `-- test_wavs
      |-- BAC009S0764W0121.wav
      |-- BAC009S0764W0122.wav
      |-- BAC009S0764W0123.wav
      `-- transcript.txt

  5 directories, 16 files


**File descriptions**:

  - ``data/lang_char``

    It contains language related files. You can find the vocabulary size in ``tokens.txt``.

  - ``exp/pretrained.pt``

      It contains pre-trained model parameters, obtained by averaging
      checkpoints from ``epoch-32.pt`` to ``epoch-64.pt``.
      Note: We have removed optimizer ``state_dict`` to reduce file size.

  - ``log``

      It contains decoding logs and decoded results.

  - ``test_wavs``

      It contains some test sound files from Aishell ``test`` dataset.

The information of the test sound files is listed below:

.. code-block:: bash

  $ soxi tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/*.wav

  Input File     : 'tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.20 = 67263 samples ~ 315.295 CDDA sectors
  File Size      : 135k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:04.12 = 65840 samples ~ 308.625 CDDA sectors
  File Size      : 132k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM


  Input File     : 'tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav'
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
  $ ./transducer_stateless_modified/pretrained.py --help

displays the help information.

It supports three decoding methods:

  - greedy search
  - beam search
  - modified beam search

.. note::

  In modified beam search, it limits the maximum number of symbols that can be
  emitted per frame to 1. To use this method, you have to ensure that your model
  has been trained with the option ``--modified-transducer-prob``. Otherwise,
  it may give you poor results.

Greedy search
^^^^^^^^^^^^^

The command to run greedy search is given below:

.. code-block:: bash


  $ cd egs/aishell/ASR
  $ ./transducer_stateless_modified/pretrained.py \
      --checkpoint ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/exp/pretrained.pt \
      --lang-dir ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char \
      --method greedy_search \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav

The output is as follows:

.. code-block::

  2022-03-03 15:35:26,531 INFO [pretrained.py:239] device: cuda:0
  2022-03-03 15:35:26,994 INFO [lexicon.py:176] Loading pre-compiled tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char/Linv.pt
  2022-03-03 15:35:27,027 INFO [pretrained.py:246] {'feature_dim': 80, 'encoder_out_dim': 512, 'subsampling_factor': 4, 'attention_dim': 512, 'nhead': 8, 'dim_feedforward': 2048, 'num_encoder_layers': 12, 'vgg_frontend': False, 'env_info': {'k2-version': '1.13', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': 'f4fefe4882bc0ae59af951da3f47335d5495ef71', 'k2-git-date': 'Thu Feb 10 15:16:02 2022', 'lhotse-version': '1.0.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.2', 'python-version': '3.8', 'icefall-git-branch': 'master', 'icefall-git-sha1': '50d2281-clean', 'icefall-git-date': 'Wed Mar 2 16:02:38 2022', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-aishell', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-multi-datasets/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-aishell/lhotse/__init__.py', 'hostname': 'de-74279-k2-train-2-0815224919-75d558775b-mmnv8', 'IP address': '10.177.72.138'}, 'sample_rate': 16000, 'checkpoint': './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/exp/pretrained.pt', 'lang_dir': PosixPath('tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char'), 'method': 'greedy_search', 'sound_files': ['./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav'], 'beam_size': 4, 'context_size': 2, 'max_sym_per_frame': 3, 'blank_id': 0, 'vocab_size': 4336}
  2022-03-03 15:35:27,027 INFO [pretrained.py:248] About to create model
  2022-03-03 15:35:36,878 INFO [pretrained.py:257] Constructing Fbank computer
  2022-03-03 15:35:36,880 INFO [pretrained.py:267] Reading sound files: ['./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav']
  2022-03-03 15:35:36,891 INFO [pretrained.py:273] Decoding started
  /ceph-fj/fangjun/open-source-2/icefall-aishell/egs/aishell/ASR/transducer_stateless_modified/conformer.py:113: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
    lengths = ((x_lens - 1) // 2 - 1) // 2
  2022-03-03 15:35:37,163 INFO [pretrained.py:320]
  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav:
  甚 至 出 现 交 易 几 乎 停 滞 的 情 况

  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav:
  一 二 线 城 市 虽 然 也 处 于 调 整 中

  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav:
  但 因 为 聚 集 了 过 多 公 共 资 源

  2022-03-03 15:35:37,163 INFO [pretrained.py:322] Decoding Done

Beam search
^^^^^^^^^^^

The command to run beam search is given below:

.. code-block:: bash


  $ cd egs/aishell/ASR

  $ ./transducer_stateless_modified/pretrained.py \
      --checkpoint ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/exp/pretrained.pt \
      --lang-dir ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char \
      --method beam_search \
      --beam-size 4 \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav

The output is as follows:

.. code-block::

  2022-03-03 15:39:09,285 INFO [pretrained.py:239] device: cuda:0
  2022-03-03 15:39:09,708 INFO [lexicon.py:176] Loading pre-compiled tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char/Linv.pt
  2022-03-03 15:39:09,759 INFO [pretrained.py:246] {'feature_dim': 80, 'encoder_out_dim': 512, 'subsampling_factor': 4, 'attention_dim': 512, 'nhead': 8, 'dim_feedforward': 2048, 'num_encoder_layers': 12, 'vgg_frontend': False, 'env_info': {'k2-version': '1.13', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': 'f4fefe4882bc0ae59af951da3f47335d5495ef71', 'k2-git-date': 'Thu Feb 10 15:16:02 2022', 'lhotse-version': '1.0.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.2', 'python-version': '3.8', 'icefall-git-branch': 'master', 'icefall-git-sha1': '50d2281-clean', 'icefall-git-date': 'Wed Mar 2 16:02:38 2022', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-aishell', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-multi-datasets/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-aishell/lhotse/__init__.py', 'hostname': 'de-74279-k2-train-2-0815224919-75d558775b-mmnv8', 'IP address': '10.177.72.138'}, 'sample_rate': 16000, 'checkpoint': './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/exp/pretrained.pt', 'lang_dir': PosixPath('tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char'), 'method': 'beam_search', 'sound_files': ['./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav'], 'beam_size': 4, 'context_size': 2, 'max_sym_per_frame': 3, 'blank_id': 0, 'vocab_size': 4336}
  2022-03-03 15:39:09,760 INFO [pretrained.py:248] About to create model
  2022-03-03 15:39:18,919 INFO [pretrained.py:257] Constructing Fbank computer
  2022-03-03 15:39:18,922 INFO [pretrained.py:267] Reading sound files: ['./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav']
  2022-03-03 15:39:18,929 INFO [pretrained.py:273] Decoding started
  /ceph-fj/fangjun/open-source-2/icefall-aishell/egs/aishell/ASR/transducer_stateless_modified/conformer.py:113: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
    lengths = ((x_lens - 1) // 2 - 1) // 2
  2022-03-03 15:39:21,046 INFO [pretrained.py:320]
  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav:
  甚 至 出 现 交 易 几 乎 停 滞 的 情 况

  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav:
  一 二 线 城 市 虽 然 也 处 于 调 整 中

  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav:
  但 因 为 聚 集 了 过 多 公 共 资 源

  2022-03-03 15:39:21,047 INFO [pretrained.py:322] Decoding Done

Modified Beam search
^^^^^^^^^^^^^^^^^^^^

The command to run modified beam search is given below:

.. code-block:: bash


  $ cd egs/aishell/ASR

  $ ./transducer_stateless_modified/pretrained.py \
      --checkpoint ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/exp/pretrained.pt \
      --lang-dir ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char \
      --method modified_beam_search \
      --beam-size 4 \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav \
      ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav

The output is as follows:

.. code-block::

  2022-03-03 15:41:23,319 INFO [pretrained.py:239] device: cuda:0
  2022-03-03 15:41:23,798 INFO [lexicon.py:176] Loading pre-compiled tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char/Linv.pt
  2022-03-03 15:41:23,831 INFO [pretrained.py:246] {'feature_dim': 80, 'encoder_out_dim': 512, 'subsampling_factor': 4, 'attention_dim': 512, 'nhead': 8, 'dim_feedforward': 2048, 'num_encoder_layers': 12, 'vgg_frontend': False, 'env_info': {'k2-version': '1.13', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': 'f4fefe4882bc0ae59af951da3f47335d5495ef71', 'k2-git-date': 'Thu Feb 10 15:16:02 2022', 'lhotse-version': '1.0.0.dev+missing.version.file', 'torch-cuda-available': True, 'torch-cuda-version': '10.2', 'python-version': '3.8', 'icefall-git-branch': 'master', 'icefall-git-sha1': '50d2281-clean', 'icefall-git-date': 'Wed Mar 2 16:02:38 2022', 'icefall-path': '/ceph-fj/fangjun/open-source-2/icefall-aishell', 'k2-path': '/ceph-fj/fangjun/open-source-2/k2-multi-datasets/k2/python/k2/__init__.py', 'lhotse-path': '/ceph-fj/fangjun/open-source-2/lhotse-aishell/lhotse/__init__.py', 'hostname': 'de-74279-k2-train-2-0815224919-75d558775b-mmnv8', 'IP address': '10.177.72.138'}, 'sample_rate': 16000, 'checkpoint': './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/exp/pretrained.pt', 'lang_dir': PosixPath('tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/data/lang_char'), 'method': 'modified_beam_search', 'sound_files': ['./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav'], 'beam_size': 4, 'context_size': 2, 'max_sym_per_frame': 3, 'blank_id': 0, 'vocab_size': 4336}
  2022-03-03 15:41:23,831 INFO [pretrained.py:248] About to create model
  2022-03-03 15:41:32,214 INFO [pretrained.py:257] Constructing Fbank computer
  2022-03-03 15:41:32,215 INFO [pretrained.py:267] Reading sound files: ['./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav', './tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav']
  2022-03-03 15:41:32,220 INFO [pretrained.py:273] Decoding started
  /ceph-fj/fangjun/open-source-2/icefall-aishell/egs/aishell/ASR/transducer_stateless_modified/conformer.py:113: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
    lengths = ((x_lens - 1) // 2 - 1) // 2
  /ceph-fj/fangjun/open-source-2/icefall-aishell/egs/aishell/ASR/transducer_stateless_modified/beam_search.py:402: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
    topk_hyp_indexes = topk_indexes // logits.size(-1)
  2022-03-03 15:41:32,583 INFO [pretrained.py:320]
  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0121.wav:
  甚 至 出 现 交 易 几 乎 停 滞 的 情 况

  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0122.wav:
  一 二 线 城 市 虽 然 也 处 于 调 整 中

  ./tmp/icefall-aishell-transducer-stateless-modified-2022-03-01/test_wavs/BAC009S0764W0123.wav:
  但 因 为 聚 集 了 过 多 公 共 资 源

  2022-03-03 15:41:32,583 INFO [pretrained.py:322] Decoding Done

Colab notebook
--------------

We provide a colab notebook for this recipe showing how to use a pre-trained model to
transcribe sound files.

|aishell asr stateless modified transducer colab notebook|

.. |aishell asr stateless modified transducer colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/12jpTxJB44vzwtcmJl2DTdznW0OawPb9H?usp=sharing
