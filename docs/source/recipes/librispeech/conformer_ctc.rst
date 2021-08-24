Confromer CTC
=============

This tutorial shows you how to run a conformer ctc model
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.


.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe.


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

  $ cd egs/yesno/ASR
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
    960 hours. Otherwise, the training part uses only 100 hours subset.

    .. CAUTION::

      The training set is perturbed by two different speeds:
      one with a value 0.9 and the other is 1.1.
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
      Mult-GPU multi-machine DDP training will be added later.

  - ``--max-duration``

    It specifies number of seconds over all utterances in a
    batch, before **padding**.
    If you encounter CUDA OOM, please reduce it. For instance, if
    your are using V100 NVIDIA GPU, we recommend you to set it to ``200``.

    .. HINT::

      Due to padding, the number of seconds of all utterances in a
      batch will usually be larger than ``--max-duration``.

      A large value for ``--max-duration`` may cause OOM during training,
      while a small value may increase the training time. You have to
      tune it.


Pre-configured options
~~~~~~~~~~~~~~~~~~~~~~

There are some training options, e.g., learning rate,
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

Pre-trained Model
-----------------

