Export model.state_dict()
=========================

When to use it
--------------

During model training, we save checkpoints periodically to disk.

A checkpoint contains the following information:

  - ``model.state_dict()``
  - ``optimizer.state_dict()``
  - and some other information related to training

When we need to resume the training process from some point, we need a checkpoint.
However, if we want to publish the model for inference, then only
``model.state_dict()`` is needed. In this case, we need to strip all other information
except ``model.state_dict()`` to reduce the file size of the published model.

How to export
-------------

Every recipe contains a file ``export.py`` that you can use to
export ``model.state_dict()`` by taking some checkpoints as inputs.

.. hint::

   Each ``export.py`` contains well-documented usage information.

In the following, we use
`<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless3/export.py>`_
as an example.

.. note::

   The steps for other recipes are almost the same.

.. code-block:: bash

  cd egs/librispeech/ASR

  ./pruned_transducer_stateless3/export.py \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --tokens data/lang_bpe_500/tokens.txt \
    --epoch 20 \
    --avg 10

will generate a file ``pruned_transducer_stateless3/exp/pretrained.pt``, which
is a dict containing ``{"model": model.state_dict()}`` saved by ``torch.save()``.

How to use the exported model
-----------------------------

For each recipe, we provide pretrained models hosted on huggingface.
You can find links to pretrained models in ``RESULTS.md`` of each dataset.

In the following, we demonstrate how to use the pretrained model from
`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>`_.

.. code-block:: bash

   cd egs/librispeech/ASR

   git lfs install
   git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

After cloning the repo with ``git lfs``, you will find several files in the folder
``icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp``
that have a prefix ``pretrained-``. Those files contain ``model.state_dict()``
exported by the above ``export.py``.

In each recipe, there is also a file ``pretrained.py``, which can use
``pretrained-xxx.pt`` to decode waves. The following is an example:

.. code-block:: bash

   cd egs/librispeech/ASR

   ./pruned_transducer_stateless3/pretrained.py \
      --checkpoint ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/pretrained-iter-1224000-avg-14.pt \
      --tokens ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/tokens.txt \
      --method greedy_search \
      ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0001.wav \
      ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0002.wav

The above commands show how to use the exported model with ``pretrained.py`` to
decode multiple sound files. Its output is given as follows for reference:

.. literalinclude:: ./code/export-model-state-dict-pretrained-out.txt

Use the exported model to run decode.py
---------------------------------------

When we publish the model, we always note down its WERs on some test
dataset in ``RESULTS.md``. This section describes how to use the
pretrained model to reproduce the WER.

.. code-block:: bash

   cd egs/librispeech/ASR
   git lfs install
   git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

   cd icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp
   ln -s pretrained-iter-1224000-avg-14.pt epoch-9999.pt
   cd ../..

We create a symlink with name ``epoch-9999.pt`` to ``pretrained-iter-1224000-avg-14.pt``,
so that we can pass ``--epoch 9999 --avg 1`` to ``decode.py`` in the following
command:

.. code-block:: bash

  ./pruned_transducer_stateless3/decode.py \
      --epoch 9999 \
      --avg 1 \
      --exp-dir ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp \
      --lang-dir ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500 \
      --max-duration 600 \
      --decoding-method greedy_search

You will find the decoding results in
``./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/greedy_search``.

.. caution::

   For some recipes, you also need to pass ``--use-averaged-model False``
   to ``decode.py``. The reason is that the exported pretrained model is already
   the averaged one.

.. hint::

   Before running ``decode.py``, we assume that you have already run
   ``prepare.sh`` to prepare the test dataset.
