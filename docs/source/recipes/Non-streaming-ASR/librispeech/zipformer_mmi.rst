Zipformer MMI
===============

.. hint::

   Please scroll down to the bottom of this page to find download links
   for pretrained models if you don't want to train a model from scratch.


This tutorial shows you how to train an Zipformer MMI model
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.

We use LF-MMI to compute the loss.

.. note::

   You can find the document about LF-MMI training at the following address:

   `<https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pdf/LF-MMI-training-and-decoding-in-k2-Part-I.pdf>`_


Data preparation
----------------

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./prepare.sh

The script ``./prepare.sh`` handles the data preparation for you, **automagically**.
All you need to do is to run it.

.. note::

   We encourage you to read ``./prepare.sh``.

The data preparation contains several stages. You can use the following two
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

.. hint::

  If you have pre-downloaded the `LibriSpeech <https://www.openslr.org/12>`_
  dataset and the `musan <http://www.openslr.org/17/>`_ dataset, say,
  they are saved in ``/tmp/LibriSpeech`` and ``/tmp/musan``, you can modify
  the ``dl_dir`` variable in ``./prepare.sh`` to point to ``/tmp`` so that
  ``./prepare.sh`` won't re-download them.

.. note::

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

For stability, it uses CTC loss for model warm-up and then switches to MMI loss.

Configurable options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./zipformer_mmi/train.py --help

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
    ``./zipformer_mmi/train.py --num-epochs 30`` trains for 30 epochs
    and generates ``epoch-1.pt``, ``epoch-2.pt``, ..., ``epoch-30.pt``
    in the folder ``./zipformer_mmi/exp``.

  - ``--start-epoch``

    It's used to resume training.
    ``./zipformer_mmi/train.py --start-epoch 10`` loads the
    checkpoint ``./zipformer_mmi/exp/epoch-9.pt`` and starts
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
          $ ./zipformer_mmi/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ ./zipformer_mmi/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ export CUDA_VISIBLE_DEVICES="3"
          $ ./zipformer_mmi/train.py --world-size 1

    .. caution::

      Only multi-GPU single-machine DDP training is implemented at present.
      Multi-GPU multi-machine DDP training will be added later.

  - ``--max-duration``

    It specifies the number of seconds over all utterances in a
    batch, before **padding**.
    If you encounter CUDA OOM, please reduce it.

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
`zipformer_mmi/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer_mmi/train.py>`_

You don't need to change these pre-configured parameters. If you really need to change
them, please modify ``./zipformer_mmi/train.py`` directly.

Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in ``zipformer_mmi/exp``.
You will find the following files in that directory:

  - ``epoch-1.pt``, ``epoch-2.pt``, ...

    These are checkpoint files saved at the end of each epoch, containing model
    ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./zipformer_mmi/train.py --start-epoch 11

  - ``checkpoint-436000.pt``, ``checkpoint-438000.pt``, ...

    These are checkpoint files saved every ``--save-every-n`` batches,
    containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``checkpoint-436000``, you can use:

      .. code-block:: bash

        $ ./zipformer_mmi/train.py --start-batch 436000

  - ``tensorboard/``

    This folder contains tensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd zipformer_mmi/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "Zipformer MMI training for LibriSpeech with icefall"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/xyOZUKpEQm62HBIlUD4uPA/

    Note there is a URL in the above output. Click it and you will see
    tensorboard.

  .. hint::

    If you don't have access to google, you can use the following command
    to view the tensorboard log locally:

      .. code-block:: bash

        cd zipformer_mmi/exp/tensorboard
        tensorboard --logdir . --port 6008

    It will print the following message:

      .. code-block::

        Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
        TensorBoard 2.8.0 at http://localhost:6008/ (Press CTRL+C to quit)

    Now start your browser and go to `<http://localhost:6008>`_ to view the tensorboard
    logs.


  - ``log/log-train-xxxx``

    It is the detailed training log in text format, same as the one
    you saw printed to the console during training.

Usage example
~~~~~~~~~~~~~

You can use the following command to start the training using 4 GPUs:

.. code-block:: bash

  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  ./zipformer_mmi/train.py \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --full-libri 1 \
    --exp-dir zipformer_mmi/exp \
    --max-duration 500 \
    --use-fp16 1 \
    --num-workers 2

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. hint::

   There are two kinds of checkpoints:

    - (1) ``epoch-1.pt``, ``epoch-2.pt``, ..., which are saved at the end
      of each epoch. You can pass ``--epoch`` to
      ``zipformer_mmi/decode.py`` to use them.

    - (2) ``checkpoints-436000.pt``, ``epoch-438000.pt``, ..., which are saved
      every ``--save-every-n`` batches. You can pass ``--iter`` to
      ``zipformer_mmi/decode.py`` to use them.

    We suggest that you try both types of checkpoints and choose the one
    that produces the lowest WERs.

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./zipformer_mmi/decode.py --help

shows the options for decoding.

The following shows the example using ``epoch-*.pt``:

.. code-block:: bash

  for m in nbest nbest-rescoring-LG nbest-rescoring-3-gram nbest-rescoring-4-gram; do
    ./zipformer_mmi/decode.py \
      --epoch 30 \
      --avg 10 \
      --exp-dir ./zipformer_mmi/exp/ \
      --max-duration 100 \
      --lang-dir data/lang_bpe_500 \
      --nbest-scale 1.2 \
      --hp-scale 1.0 \
      --decoding-method $m
  done


Export models
-------------

`zipformer_mmi/export.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer_mmi/export.py>`_ supports exporting checkpoints from ``zipformer_mmi/exp`` in the following ways.

Export ``model.state_dict()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoints saved by ``zipformer_mmi/train.py`` also include
``optimizer.state_dict()``. It is useful for resuming training. But after training,
we are interested only in ``model.state_dict()``. You can use the following
command to extract ``model.state_dict()``.

.. code-block:: bash

  ./zipformer_mmi/export.py \
    --exp-dir ./zipformer_mmi/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9 \
    --jit 0

It will generate a file ``./zipformer_mmi/exp/pretrained.pt``.

.. hint::

   To use the generated ``pretrained.pt`` for ``zipformer_mmi/decode.py``,
   you can run:

   .. code-block:: bash

      cd zipformer_mmi/exp
      ln -s pretrained epoch-9999.pt

   And then pass ``--epoch 9999 --avg 1 --use-averaged-model 0`` to
   ``./zipformer_mmi/decode.py``.

To use the exported model with ``./zipformer_mmi/pretrained.py``, you
can run:

.. code-block:: bash

  ./zipformer_mmi/pretrained.py \
    --checkpoint ./zipformer_mmi/exp/pretrained.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method 1best \
    /path/to/foo.wav \
    /path/to/bar.wav

Export model using ``torch.jit.script()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./zipformer_mmi/export.py \
    --exp-dir ./zipformer_mmi/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9 \
    --jit 1

It will generate a file ``cpu_jit.pt`` in the given ``exp_dir``. You can later
load it by ``torch.jit.load("cpu_jit.pt")``.

Note ``cpu`` in the name ``cpu_jit.pt`` means the parameters when loaded into Python
are on CPU. You can use ``to("cuda")`` to move them to a CUDA device.

To use the generated files with ``./zipformer_mmi/jit_pretrained.py``:

.. code-block:: bash

  ./zipformer_mmi/jit_pretrained.py \
    --nn-model-filename ./zipformer_mmi/exp/cpu_jit.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method 1best \
    /path/to/foo.wav \
    /path/to/bar.wav

Download pretrained models
--------------------------

If you don't want to train from scratch, you can download the pretrained models
by visiting the following links:

  - `<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-mmi-2022-12-08>`_

  See `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_
  for the details of the above pretrained models
