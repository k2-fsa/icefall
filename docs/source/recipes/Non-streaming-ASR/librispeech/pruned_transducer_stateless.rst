Pruned transducer statelessX
============================

This tutorial shows you how to run a conformer transducer model
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset.

.. Note::

   The tutorial is suitable for `pruned_transducer_stateless <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless>`_,
   `pruned_transducer_stateless2 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless2>`_,
   `pruned_transducer_stateless4 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless4>`_,
   `pruned_transducer_stateless5 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless5>`_,
   We will take pruned_transducer_stateless4 as an example in this tutorial.

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe.

.. hint::

   Please scroll down to the bottom of this page to find download links
   for pretrained models if you don't want to train a model from scratch.


We use pruned RNN-T to compute the loss.

.. note::

   You can find the paper about pruned RNN-T at the following address:

   `<https://arxiv.org/abs/2206.13236>`_

The transducer model consists of 3 parts:

  - Encoder, a.k.a, the transcription network. We use a Conformer model (the reworked version by Daniel Povey)
  - Decoder, a.k.a, the prediction network. We use a stateless model consisting of
    ``nn.Embedding`` and ``nn.Conv1d``
  - Joiner, a.k.a, the joint network.

.. caution::

   Contrary to the conventional RNN-T models, we use a stateless decoder.
   That is, it has no recurrent connections.


Data preparation
----------------

.. hint::

   The data preparation is the same as other recipes on LibriSpeech dataset,
   if you have finished this step, you can skip to ``Training`` directly.

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
  $ ./pruned_transducer_stateless4/train.py --help


shows you the training options that can be passed from the commandline.
The following options are used quite often:

  - ``--exp-dir``

    The directory to save checkpoints, training logs and tensorboard.

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
    ``./pruned_transducer_stateless4/train.py --num-epochs 30`` trains for 30 epochs
    and generates ``epoch-1.pt``, ``epoch-2.pt``, ..., ``epoch-30.pt``
    in the folder ``./pruned_transducer_stateless4/exp``.

  - ``--start-epoch``

    It's used to resume training.
    ``./pruned_transducer_stateless4/train.py --start-epoch 10`` loads the
    checkpoint ``./pruned_transducer_stateless4/exp/epoch-9.pt`` and starts
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
          $ ./pruned_transducer_stateless4/train.py --world-size 2

      **Use case 2**: You have 4 GPUs and you want to use all of them
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ ./pruned_transducer_stateless4/train.py --world-size 4

      **Use case 3**: You have 4 GPUs but you only want to use GPU 3
      for training. You can do the following:

        .. code-block:: bash

          $ cd egs/librispeech/ASR
          $ export CUDA_VISIBLE_DEVICES="3"
          $ ./pruned_transducer_stateless4/train.py --world-size 1

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

  - ``--use-fp16``

    If it is True, the model will train with half precision, from our experiment
    results, by using half precision you can train with two times larger ``--max-duration``
    so as to get almost 2X speed up.


Pre-configured options
~~~~~~~~~~~~~~~~~~~~~~

There are some training options, e.g., number of encoder layers,
encoder dimension, decoder dimension, number of warmup steps etc,
that are not passed from the commandline.
They are pre-configured by the function ``get_params()`` in
`pruned_transducer_stateless4/train.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless4/train.py>`_

You don't need to change these pre-configured parameters. If you really need to change
them, please modify ``./pruned_transducer_stateless4/train.py`` directly.


.. NOTE::

  The options for `pruned_transducer_stateless5 <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless5/train.py>`_ are a little different from
  other recipes. It allows you to configure ``--num-encoder-layers``, ``--dim-feedforward``, ``--nhead``, ``--encoder-dim``, ``--decoder-dim``, ``--joiner-dim`` from commandline, so that you can train models with different size with pruned_transducer_stateless5.


Training logs
~~~~~~~~~~~~~

Training logs and checkpoints are saved in ``--exp-dir`` (e.g. ``pruned_transducer_stateless4/exp``.
You will find the following files in that directory:

  - ``epoch-1.pt``, ``epoch-2.pt``, ...

    These are checkpoint files saved at the end of each epoch, containing model
    ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``epoch-10.pt``, you can use:

      .. code-block:: bash

        $ ./pruned_transducer_stateless4/train.py --start-epoch 11

  - ``checkpoint-436000.pt``, ``checkpoint-438000.pt``, ...

    These are checkpoint files saved every ``--save-every-n`` batches,
    containing model ``state_dict`` and optimizer ``state_dict``.
    To resume training from some checkpoint, say ``checkpoint-436000``, you can use:

      .. code-block:: bash

        $ ./pruned_transducer_stateless4/train.py --start-batch 436000

  - ``tensorboard/``

    This folder contains tensorBoard logs. Training loss, validation loss, learning
    rate, etc, are recorded in these logs. You can visualize them by:

      .. code-block:: bash

        $ cd pruned_transducer_stateless4/exp/tensorboard
        $ tensorboard dev upload --logdir . --description "pruned transducer training for LibriSpeech with icefall"

    It will print something like below:

      .. code-block::

        TensorFlow installation not found - running with reduced feature set.
        Upload started and will continue reading any new data as it's added to the logdir.

        To stop uploading, press Ctrl-C.

        New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/QOGSPBgsR8KzcRMmie9JGw/

        [2022-11-20T15:50:50] Started scanning logdir.
        Uploading 4468 scalars...
        [2022-11-20T15:53:02] Total uploaded: 210171 scalars, 0 tensors, 0 binary objects
        Listening for new data in logdir...

    Note there is a URL in the above output. Click it and you will see
    the following screenshot:

      .. figure:: images/librispeech-pruned-transducer-tensorboard-log.jpg
         :width: 600
         :alt: TensorBoard screenshot
         :align: center
         :target: https://tensorboard.dev/experiment/QOGSPBgsR8KzcRMmie9JGw/

         TensorBoard screenshot.

  .. hint::

    If you don't have access to google, you can use the following command
    to view the tensorboard log locally:

      .. code-block:: bash

        cd pruned_transducer_stateless4/exp/tensorboard
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

You can use the following command to start the training using 6 GPUs:

.. code-block:: bash

  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
  ./pruned_transducer_stateless4/train.py \
     --world-size 6 \
     --num-epochs 30 \
     --start-epoch 1 \
     --exp-dir pruned_transducer_stateless4/exp \
     --full-libri 1 \
     --max-duration 300


Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

.. hint::

   There are two kinds of checkpoints:

    - (1) ``epoch-1.pt``, ``epoch-2.pt``, ..., which are saved at the end
      of each epoch. You can pass ``--epoch`` to
      ``pruned_transducer_stateless4/decode.py`` to use them.

    - (2) ``checkpoints-436000.pt``, ``epoch-438000.pt``, ..., which are saved
      every ``--save-every-n`` batches. You can pass ``--iter`` to
      ``pruned_transducer_stateless4/decode.py`` to use them.

    We suggest that you try both types of checkpoints and choose the one
    that produces the lowest WERs.

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ ./pruned_transducer_stateless4/decode.py --help

shows the options for decoding.

The following shows two examples (for two types of checkpoints):

.. code-block:: bash

  for m in greedy_search fast_beam_search modified_beam_search; do
    for epoch in 25 20; do
      for avg in 7 5 3 1; do
        ./pruned_transducer_stateless4/decode.py \
          --epoch $epoch \
          --avg $avg \
          --exp-dir pruned_transducer_stateless4/exp \
          --max-duration 600 \
          --decoding-method $m
      done
    done
  done


.. code-block:: bash

  for m in greedy_search fast_beam_search modified_beam_search; do
    for iter in 474000; do
      for avg in 8 10 12 14 16 18; do
        ./pruned_transducer_stateless4/decode.py \
          --iter $iter \
          --avg $avg \
          --exp-dir pruned_transducer_stateless4/exp \
          --max-duration 600 \
          --decoding-method $m
      done
    done
  done


.. Note::

  Supporting decoding methods are as follows:

    - ``greedy_search`` : It takes the symbol with largest posterior probability
      of each frame as the decoding result.

    - ``beam_search`` :  It implements Algorithm 1 in https://arxiv.org/pdf/1211.3711.pdf and
      `espnet/nets/beam_search_transducer.py <https://github.com/espnet/espnet/blob/master/espnet/nets/beam_search_transducer.py#L247>`_
      is used as a reference. Basicly, it keeps topk states for each frame, and expands the kept states with their own contexts to
      next frame.

    - ``modified_beam_search`` : It implements the same algorithm as ``beam_search`` above, but it
      runs in batch mode with ``--max-sym-per-frame=1`` being hardcoded.

    - ``fast_beam_search`` : It implements graph composition between the output ``log_probs`` and
      given ``FSAs``. It is hard to describe the details in several lines of texts, you can read
      our paper in https://arxiv.org/pdf/2211.00484.pdf or our `rnnt decode code in k2 <https://github.com/k2-fsa/k2/blob/master/k2/csrc/rnnt_decode.h>`_. ``fast_beam_search`` can decode with ``FSAs`` on GPU efficiently.

    - ``fast_beam_search_LG`` : The same as ``fast_beam_search`` above, ``fast_beam_search`` uses
      an trivial graph that has only one state, while ``fast_beam_search_LG`` uses an LG graph
      (with N-gram LM).

    - ``fast_beam_search_nbest`` : It produces the decoding results as follows:

      - (1) Use ``fast_beam_search`` to get a lattice
      - (2) Select ``num_paths`` paths from the lattice using ``k2.random_paths()``
      - (3) Unique the selected paths
      - (4) Intersect the selected paths with the lattice and compute the
            shortest path from the intersection result
      - (5) The path with the largest score is used as the decoding output.

    - ``fast_beam_search_nbest_LG`` : It implements same logic as ``fast_beam_search_nbest``, the
      only difference is that it uses ``fast_beam_search_LG`` to generate the lattice.


Export Model
------------

`pruned_transducer_stateless4/export.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless4/export.py>`_ supports exporting checkpoints from ``pruned_transducer_stateless4/exp`` in the following ways.

Export ``model.state_dict()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoints saved by ``pruned_transducer_stateless4/train.py`` also include
``optimizer.state_dict()``. It is useful for resuming training. But after training,
we are interested only in ``model.state_dict()``. You can use the following
command to extract ``model.state_dict()``.

.. code-block:: bash

  # Assume that --epoch 25 --avg 3 produces the smallest WER
  # (You can get such information after running ./pruned_transducer_stateless4/decode.py)

  epoch=25
  avg=3

  ./pruned_transducer_stateless4/export.py \
    --exp-dir ./pruned_transducer_stateless4/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch $epoch \
    --avg  $avg

It will generate a file ``./pruned_transducer_stateless4/exp/pretrained.pt``.

.. hint::

   To use the generated ``pretrained.pt`` for ``pruned_transducer_stateless4/decode.py``,
   you can run:

   .. code-block:: bash

      cd pruned_transducer_stateless4/exp
      ln -s pretrained.pt epoch-999.pt

   And then pass ``--epoch 999 --avg 1 --use-averaged-model 0`` to
   ``./pruned_transducer_stateless4/decode.py``.

To use the exported model with ``./pruned_transducer_stateless4/pretrained.py``, you
can run:

.. code-block:: bash

  ./pruned_transducer_stateless4/pretrained.py \
    --checkpoint ./pruned_transducer_stateless4/exp/pretrained.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method greedy_search \
    /path/to/foo.wav \
    /path/to/bar.wav


Export model using ``torch.jit.script()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  ./pruned_transducer_stateless4/export.py \
    --exp-dir ./pruned_transducer_stateless4/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 25 \
    --avg 3 \
    --jit 1

It will generate a file ``cpu_jit.pt`` in the given ``exp_dir``. You can later
load it by ``torch.jit.load("cpu_jit.pt")``.

Note ``cpu`` in the name ``cpu_jit.pt`` means the parameters when loaded into Python
are on CPU. You can use ``to("cuda")`` to move them to a CUDA device.

.. NOTE::

   You will need this ``cpu_jit.pt`` when deploying with Sherpa framework.


Download pretrained models
--------------------------

If you don't want to train from scratch, you can download the pretrained models
by visiting the following links:

  - `pruned_transducer_stateless <https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless-2022-03-12>`_

  - `pruned_transducer_stateless2 <https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless2-2022-04-29>`_

  - `pruned_transducer_stateless4 <https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless4-2022-06-03>`_

  - `pruned_transducer_stateless5 <https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless5-2022-07-07>`_

  See `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>`_
  for the details of the above pretrained models


Deploy with Sherpa
------------------

Please see `<https://k2-fsa.github.io/sherpa/python/offline_asr/conformer/librispeech.html#>`_
for how to deploy the models in ``sherpa``.
