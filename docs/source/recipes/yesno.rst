yesno
=====

This page shows you how to run the ``yesno`` recipe.

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  You **don't** need a **GPU** to run this recipe. It can be run on a **CPU**.
  The training time takes less than 30 **seconds** and you will get
  the following WER::

    [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]

Data preparation
----------------

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ ./prepare.sh

The script ``./prepare.sh`` handles the data preparation for you, automagically.
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

    These are checkpoint files, containing model parameters and optimizer ``state_dict``.
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

      .. figure:: images/yesno-tdnn-tensorboard-log.png
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/yKUbhb5wRmOSXYkId1z9eg/

         TensorBoard screenshot.

  - ``log/log-train-xxxx``

    It is the detailed training log in text format, same as the one
    you saw printed to the console during training.


To see available training options, you can use:

.. code-block:: bash

  $ ./tdnn/train.py --help

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

Colab notebook
--------------

We do provide a colab notebook for this recipe.

|yesno colab notebook|

.. |yesno colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing



Use a pre-trained model
-----------------------

TODO
