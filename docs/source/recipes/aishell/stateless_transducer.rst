Stateless Transducer
====================

This tutorial shows you how to do transducer training in ``icefall``.

.. HINT::

  Instead of using RNN-T or RNN transducer, we only use transducer
  here. As you will see, there are no RNNs in the model.

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
    - A Conv1d layer with a default kernel size 2

- **Joiner**: It consists of a ``nn.tanh()`` and a ``nn.Linear()``.

.. Caution::

  The decoder is stateless and very simple. It is borrowed from
  `<https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419>`_
  (Rnn-Transducer with Stateless Prediction Network)

  We make one modification to it: Place a Conv1d layer right after
  the embedding layer.

When using Chinese characters as modelling unit, whose vocabulary size
is 4335 in this specific dataset,
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

  You can use ``./prepare.sh``, though it will generates FSTs that
  are not used in transducer traning.

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

    It is used for multi-GPU single-machine DDP training.

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

      Only multi-GPU single-machine DDP training is implemented at present.
      There is an on-going PR `<https://github.com/k2-fsa/icefall/pull/63>`
      that adds support for multi-GPU multi-machine DDP training.

  - ``--max-duration``

    It specifies the number of seconds over all utterances in a
    batch, before **padding**.
    If you encounter CUDA OOM, please reduce it. For instance, if
    your are using V100 NVIDIA GPU with 32 GB RAM, we recommend you
    to set it to ``300``.

    .. HINT::

      Due to padding, the number of seconds of all utterances in a
      batch will usually be larger than ``--max-duration``.

      A larger value for ``--max-duration`` may cause OOM during training,
      while a smaller value may increase the training time. You have to
      tune it.

  - ``--lr-factor``

    It contrals the learning rate. If you use single GPU training, you
    may want to use a small value for it. If you use multiple GPUs for training,
    you may increase it.

  - ``--context-size``

    It specifies the kernel size in the decoder. Default value 2 means it
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
