LSTM Transducer
===============

.. hint::

   Please scroll down to the bottom of this page to find download links
   for pretrained models if you don't want to train a model from scratch.


This tutorial shows you how to train an LSTM transducer model
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.

We use pruned RNN-T to compute the loss.

.. note::

   You can find the paper about pruned RNN-T at the following address:

   `<https://arxiv.org/abs/2206.13236>`_

The transducer model consists of 3 parts:

  - Encoder, a.k.a, the transcription network. We use an LSTM model
  - Decoder, a.k.a, the prediction network. We use a stateless model consisting of
    ``nn.Embedding`` and ``nn.Conv1d``
  - Joiner, a.k.a, the joint network.

.. caution::

   Contrary to the conventional RNN-T models, we use a stateless decoder.
   That is, it has no recurrent connections.

.. hint::

   Since the encoder model is an LSTM, not Transformer/Conformer, the
   resulting model is suitable for streaming/online ASR.


Which model to use
------------------

Currently, there are two folders about LSTM stateless transducer training:

  - ``(1)`` `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless>`_

    This recipe uses only LibriSpeech during training.

  - ``(2)`` `<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2>`_

    This recipe uses GigaSpeech + LibriSpeech during training.

``(1)`` and ``(2)`` use the same model architecture. The only difference is that ``(2)`` supports
multi-dataset. Since ``(2)`` uses more data, it has a lower WER than ``(1)`` but it needs
more training time.

We use ``lstm_transducer_stateless2`` as an example below.

.. note::

   You need to download the `GigaSpeech <https://github.com/SpeechColab/GigaSpeech>`_ dataset
   to run ``(2)``. If you have only ``LibriSpeech`` dataset available, feel free to use ``(1)``.

Data preparation
----------------

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./prepare.sh

  # If you use (1), you can **skip** the following command
  $ ./prepare_giga_speech.sh

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

Configurable options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./lstm_transducer_stateless2/train.py --help

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
    ``./lstm_transducer_stateless2/train.py --num-epochs 30`` trains for 30 epochs
    and generates ``epoch-1.pt``, ``epoch-2.pt``, ..., ``epoch-30.pt``
    in the folder ``./lstm_transducer_stateless2/exp``.

  - ``--start-epoch``

    It's used to resume training.
    ``./lstm_transducer_stateless2/train.py --start-epoch 10`` loads the
    checkpoint ``./lstm_transducer_stateless2/exp/epoch-9.pt`` and starts
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
          $ ./lstm_transducer_stateless2/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ ./lstm_transducer_stateless2/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ export CUDA_VISIBLE_DEVICES="3"
          $ ./lstm_transducer_stateless2/train.py --world-size 1

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

  - ``--giga-prob``

    The probability to select a batch from the ``GigaSpeech`` dataset.
    Note: It is available only for ``(2)``.

Pre-configured options
~~~~~~~~~~~~~~~~~~~~~~

There are some training options, e.g., weight decay,
number of warmup steps, results dir, etc,
that are not passed from the commandline.
They are pre-configured by the function ``get_params()`` in
`lstm_transducer_stateless2/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/lstm_transducer_stateless2/train.py>`_

You don't need to change these pre-configured parameters. If you really need to change
them, please modify ``./lstm_transducer_stateless2/train.py`` directly.

Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in ``lstm_transducer_stateless2/exp``.
You will find the following files in that directory:

  - ``epoch-1.pt``, ``epoch-2.pt``, ...

    These are checkpoint files saved at the end of each epoch, containing model
    ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./lstm_transducer_stateless2/train.py --start-epoch 11

  - ``checkpoint-436000.pt``, ``checkpoint-438000.pt``, ...

    These are checkpoint files saved every ``--save-every-n`` batches,
    containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``checkpoint-436000``, you can use:

      .. code-block:: bash

        $ ./lstm_transducer_stateless2/train.py --start-batch 436000

  - ``tensorboard/``

    This folder contains tensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd lstm_transducer_stateless2/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "LSTM transducer training for LibriSpeech with icefall"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/cj2vtPiwQHKN9Q1tx6PTpg/

        [2022-09-20T15:50:50] Started scanning logdir.
        Uploading 4468 scalars...
        [2022-09-20T15:53:02] Total uploaded: 210171 scalars, 0 tensors, 0 binary objects
        Listening for new data in logdir...

    Note there is a URL in the above output. Click it and you will see
    the following screenshot:

      .. figure:: images/librispeech-lstm-transducer-tensorboard-log.png
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/lzGnETjwRxC3yghNMd4kPw/

         TensorBoard screenshot.

  .. hint::

    If you don't have access to google, you can use the following command
    to view the tensorboard log locally:

      .. code-block:: bash

        cd lstm_transducer_stateless2/exp/tensorboard
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

You can use the following command to start the training using 8 GPUs:

.. code-block:: bash

  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  ./lstm_transducer_stateless2/train.py \
    --world-size 8 \
    --num-epochs 35 \
    --start-epoch 1 \
    --full-libri 1 \
    --exp-dir lstm_transducer_stateless2/exp \
    --max-duration 500 \
    --use-fp16 0 \
    --lr-epochs 10 \
    --num-workers 2 \
    --giga-prob 0.9

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. hint::

   There are two kinds of checkpoints:

    - (1) ``epoch-1.pt``, ``epoch-2.pt``, ..., which are saved at the end
      of each epoch. You can pass ``--epoch`` to
      ``lstm_transducer_stateless2/decode.py`` to use them.

    - (2) ``checkpoints-436000.pt``, ``epoch-438000.pt``, ..., which are saved
      every ``--save-every-n`` batches. You can pass ``--iter`` to
      ``lstm_transducer_stateless2/decode.py`` to use them.

    We suggest that you try both types of checkpoints and choose the one
    that produces the lowest WERs.

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./lstm_transducer_stateless2/decode.py --help

shows the options for decoding.

The following shows two examples:

.. code-block:: bash

  for m in greedy_search fast_beam_search modified_beam_search; do
    for epoch in 17; do
      for avg in 1 2; do
        ./lstm_transducer_stateless2/decode.py \
          --epoch $epoch \
          --avg $avg \
          --exp-dir lstm_transducer_stateless2/exp \
          --max-duration 600 \
          --num-encoder-layers 12 \
          --rnn-hidden-size 1024 \
          --decoding-method $m \
          --use-averaged-model True \
          --beam 4 \
          --max-contexts 4 \
          --max-states 8 \
          --beam-size 4
      done
    done
  done


.. code-block:: bash

  for m in greedy_search fast_beam_search modified_beam_search; do
    for iter in 474000; do
      for avg in 8 10 12 14 16 18; do
        ./lstm_transducer_stateless2/decode.py \
          --iter $iter \
          --avg $avg \
          --exp-dir lstm_transducer_stateless2/exp \
          --max-duration 600 \
          --num-encoder-layers 12 \
          --rnn-hidden-size 1024 \
          --decoding-method $m \
          --use-averaged-model True \
          --beam 4 \
          --max-contexts 4 \
          --max-states 8 \
          --beam-size 4
      done
    done
  done

Export models
-------------

`lstm_transducer_stateless2/export.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/lstm_transducer_stateless2/export.py>`_ supports exporting checkpoints from ``lstm_transducer_stateless2/exp`` in the following ways.

Export ``model.state_dict()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoints saved by ``lstm_transducer_stateless2/train.py`` also include
``optimizer.state_dict()``. It is useful for resuming training. But after training,
we are interested only in ``model.state_dict()``. You can use the following
command to extract ``model.state_dict()``.

.. code-block:: bash

  # Assume that --iter 468000 --avg 16 produces the smallest WER
  # (You can get such information after running ./lstm_transducer_stateless2/decode.py)

  iter=468000
  avg=16

  ./lstm_transducer_stateless2/export.py \
    --exp-dir ./lstm_transducer_stateless2/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --iter $iter \
    --avg  $avg

It will generate a file ``./lstm_transducer_stateless2/exp/pretrained.pt``.

.. hint::

   To use the generated ``pretrained.pt`` for ``lstm_transducer_stateless2/decode.py``,
   you can run:

   .. code-block:: bash

      cd lstm_transducer_stateless2/exp
      ln -s pretrained epoch-9999.pt

   And then pass ``--epoch 9999 --avg 1 --use-averaged-model 0`` to
   ``./lstm_transducer_stateless2/decode.py``.

To use the exported model with ``./lstm_transducer_stateless2/pretrained.py``, you
can run:

.. code-block:: bash

  ./lstm_transducer_stateless2/pretrained.py \
    --checkpoint ./lstm_transducer_stateless2/exp/pretrained.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method greedy_search \
    /path/to/foo.wav \
    /path/to/bar.wav

Export model using ``torch.jit.trace()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  iter=468000
  avg=16

  ./lstm_transducer_stateless2/export.py \
    --exp-dir ./lstm_transducer_stateless2/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --iter $iter \
    --avg  $avg \
    --jit-trace 1

It will generate 3 files:

  - ``./lstm_transducer_stateless2/exp/encoder_jit_trace.pt``
  - ``./lstm_transducer_stateless2/exp/decoder_jit_trace.pt``
  - ``./lstm_transducer_stateless2/exp/joiner_jit_trace.pt``

To use the generated files with ``./lstm_transducer_stateless2/jit_pretrained``:

.. code-block:: bash

  ./lstm_transducer_stateless2/jit_pretrained.py \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --encoder-model-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace.pt \
    --decoder-model-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace.pt \
    --joiner-model-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace.pt \
    /path/to/foo.wav \
    /path/to/bar.wav

.. hint::

   Please see `<https://k2-fsa.github.io/sherpa/python/streaming_asr/lstm/english/server.html>`_
   for how to use the exported models in ``sherpa``.

Download pretrained models
--------------------------

If you don't want to train from scratch, you can download the pretrained models
by visiting the following links:

  - `<https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03>`_

  - `<https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18>`_

  See `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_
  for the details of the above pretrained models

You can find more usages of the pretrained models in
`<https://k2-fsa.github.io/sherpa/python/streaming_asr/lstm/index.html>`_
