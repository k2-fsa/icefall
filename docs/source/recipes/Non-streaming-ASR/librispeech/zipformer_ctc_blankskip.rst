Zipformer CTC Blank Skip
========================

.. hint::

   Please scroll down to the bottom of this page to find download links
   for pretrained models if you don't want to train a model from scratch.


This tutorial shows you how to train a Zipformer model based on the guidance from 
a co-trained CTC model using `blank skip method <https://arxiv.org/pdf/2210.16481.pdf>`_
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.

.. note::

    We use both CTC and RNN-T loss to train. During the forward pass, the encoder output
    is first used to calculate the CTC posterior probability; then for each output frame,
    if its blank posterior is bigger than some threshold, it will be simply discarded
    from the encoder output. To prevent information loss, we also put a convolution module
    similar to the one used in conformer (referred to as “LConv”) before the frame reduction.


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

For stability, it doesn`t use blank skip method until model warm-up.

Configurable options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./pruned_transducer_stateless7_ctc_bs/train.py --help

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
    ``./pruned_transducer_stateless7_ctc_bs/train.py --num-epochs 30`` trains for 30 epochs
    and generates ``epoch-1.pt``, ``epoch-2.pt``, ..., ``epoch-30.pt``
    in the folder ``./pruned_transducer_stateless7_ctc_bs/exp``.

  - ``--start-epoch``

    It's used to resume training.
    ``./pruned_transducer_stateless7_ctc_bs/train.py --start-epoch 10`` loads the
    checkpoint ``./pruned_transducer_stateless7_ctc_bs/exp/epoch-9.pt`` and starts
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
          $ ./pruned_transducer_stateless7_ctc_bs/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ ./pruned_transducer_stateless7_ctc_bs/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ export CUDA_VISIBLE_DEVICES="3"
          $ ./pruned_transducer_stateless7_ctc_bs/train.py --world-size 1

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
`pruned_transducer_stateless7_ctc_bs/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_ctc_bs/train.py>`_

You don't need to change these pre-configured parameters. If you really need to change
them, please modify ``./pruned_transducer_stateless7_ctc_bs/train.py`` directly.

Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in ``pruned_transducer_stateless7_ctc_bs/exp``.
You will find the following files in that directory:

  - ``epoch-1.pt``, ``epoch-2.pt``, ...

    These are checkpoint files saved at the end of each epoch, containing model
    ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./pruned_transducer_stateless7_ctc_bs/train.py --start-epoch 11

  - ``checkpoint-436000.pt``, ``checkpoint-438000.pt``, ...

    These are checkpoint files saved every ``--save-every-n`` batches,
    containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``checkpoint-436000``, you can use:

      .. code-block:: bash

        $ ./pruned_transducer_stateless7_ctc_bs/train.py --start-batch 436000

  - ``tensorboard/``

    This folder contains tensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd pruned_transducer_stateless7_ctc_bs/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "Zipformer-CTC co-training using blank skip for LibriSpeech with icefall"

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

        cd pruned_transducer_stateless7_ctc_bs/exp/tensorboard
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
  ./pruned_transducer_stateless7_ctc_bs/train.py \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --full-libri 1 \
    --exp-dir pruned_transducer_stateless7_ctc_bs/exp \
    --max-duration 600 \
    --use-fp16 1

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. hint::

   There are two kinds of checkpoints:

    - (1) ``epoch-1.pt``, ``epoch-2.pt``, ..., which are saved at the end
      of each epoch. You can pass ``--epoch`` to
      ``pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py`` to use them.

    - (2) ``checkpoints-436000.pt``, ``epoch-438000.pt``, ..., which are saved
      every ``--save-every-n`` batches. You can pass ``--iter`` to
      ``pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py`` to use them.

    We suggest that you try both types of checkpoints and choose the one
    that produces the lowest WERs.

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py --help

shows the options for decoding.

The following shows the example using ``epoch-*.pt``:

.. code-block:: bash

    for m in greedy_search fast_beam_search modified_beam_search; do
        ./pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py \
            --epoch 30 \
            --avg 13 \
            --exp-dir pruned_transducer_stateless7_ctc_bs/exp \
            --max-duration 600 \
            --decoding-method $m
    done

To test CTC branch, you can use the following command:

.. code-block:: bash

    for m in ctc-decoding 1best; do
        ./pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py \
            --epoch 30 \
            --avg 13 \
            --exp-dir pruned_transducer_stateless7_ctc_bs/exp \
            --max-duration 600 \
            --decoding-method $m
    done

Export models
-------------

`pruned_transducer_stateless7_ctc_bs/export.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_ctc_bs/export.py>`_ supports exporting checkpoints from ``pruned_transducer_stateless7_ctc_bs/exp`` in the following ways.

Export ``model.state_dict()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoints saved by ``pruned_transducer_stateless7_ctc_bs/train.py`` also include
``optimizer.state_dict()``. It is useful for resuming training. But after training,
we are interested only in ``model.state_dict()``. You can use the following
command to extract ``model.state_dict()``.

.. code-block:: bash

  ./pruned_transducer_stateless7_ctc_bs/export.py \
    --exp-dir ./pruned_transducer_stateless7_ctc_bs/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 13 \
    --jit 0

It will generate a file ``./pruned_transducer_stateless7_ctc_bs/exp/pretrained.pt``.

.. hint::

   To use the generated ``pretrained.pt`` for ``pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py``,
   you can run:

   .. code-block:: bash

      cd pruned_transducer_stateless7_ctc_bs/exp
      ln -s pretrained epoch-9999.pt

   And then pass ``--epoch 9999 --avg 1 --use-averaged-model 0`` to
   ``./pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py``.

To use the exported model with ``./pruned_transducer_stateless7_ctc_bs/pretrained.py``, you
can run:

.. code-block:: bash

  ./pruned_transducer_stateless7_ctc_bs/pretrained.py \
    --checkpoint ./pruned_transducer_stateless7_ctc_bs/exp/pretrained.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method greedy_search \
    /path/to/foo.wav \
    /path/to/bar.wav

To test CTC branch using the exported model with ``./pruned_transducer_stateless7_ctc_bs/pretrained_ctc.py``:

.. code-block:: bash

  ./pruned_transducer_stateless7_ctc_bs/jit_pretrained_ctc.py \
    --checkpoint ./pruned_transducer_stateless7_ctc_bs/exp/pretrained.pt \
    --bpe-model data/lang_bpe_500/bpe.model \
    --method ctc-decoding \
    --sample-rate 16000 \
    /path/to/foo.wav \
    /path/to/bar.wav

Export model using ``torch.jit.script()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./pruned_transducer_stateless7_ctc_bs/export.py \
    --exp-dir ./pruned_transducer_stateless7_ctc_bs/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 13 \
    --jit 1

It will generate a file ``cpu_jit.pt`` in the given ``exp_dir``. You can later
load it by ``torch.jit.load("cpu_jit.pt")``.

Note ``cpu`` in the name ``cpu_jit.pt`` means the parameters when loaded into Python
are on CPU. You can use ``to("cuda")`` to move them to a CUDA device.

To use the generated files with ``./pruned_transducer_stateless7_ctc_bs/jit_pretrained.py``:

.. code-block:: bash

  ./pruned_transducer_stateless7_ctc_bs/jit_pretrained.py \
    --nn-model-filename ./pruned_transducer_stateless7_ctc_bs/exp/cpu_jit.pt \
    /path/to/foo.wav \
    /path/to/bar.wav

To test CTC branch using the generated files with ``./pruned_transducer_stateless7_ctc_bs/jit_pretrained_ctc.py``:

.. code-block:: bash

  ./pruned_transducer_stateless7_ctc_bs/jit_pretrained_ctc.py \
    --model-filename ./pruned_transducer_stateless7_ctc_bs/exp/cpu_jit.pt \
    --bpe-model data/lang_bpe_500/bpe.model \
    --method ctc-decoding \
    --sample-rate 16000 \
    /path/to/foo.wav \
    /path/to/bar.wav

Download pretrained models
--------------------------

If you don't want to train from scratch, you can download the pretrained models
by visiting the following links:

  - trained on LibriSpeech 100h: `<https://huggingface.co/yfyeung/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2022-12-14>`_
  - trained on LibriSpeech 960h: `<https://huggingface.co/yfyeung/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29>`_

  See `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_
  for the details of the above pretrained models
