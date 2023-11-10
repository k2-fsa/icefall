.. _export-model-with-torch-jit-trace:

Export model with torch.jit.trace()
===================================

In this section, we describe how to export a model via
``torch.jit.trace()``.

When to use it
--------------

If we want to use our trained model with torchscript,
we can use ``torch.jit.trace()``.

.. hint::

  See :ref:`export-model-with-torch-jit-script`
  if you want to use ``torch.jit.script()``.

How to export
-------------

We use
`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2>`_
as an example in the following.

.. code-block:: bash

    iter=468000
    avg=16

    cd egs/librispeech/ASR

    ./lstm_transducer_stateless2/export.py \
      --exp-dir ./lstm_transducer_stateless2/exp \
      --bpe-model data/lang_bpe_500/bpe.model \
      --iter $iter \
      --avg  $avg \
      --jit-trace 1

It will generate three files inside ``lstm_transducer_stateless2/exp``:

  - ``encoder_jit_trace.pt``
  - ``decoder_jit_trace.pt``
  - ``joiner_jit_trace.pt``

You can use
`<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/lstm_transducer_stateless2/jit_pretrained.py>`_
to decode sound files with the following commands:

.. code-block:: bash

    cd egs/librispeech/ASR
    ./lstm_transducer_stateless2/jit_pretrained.py \
      --bpe-model ./data/lang_bpe_500/bpe.model \
      --encoder-model-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace.pt \
      --decoder-model-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace.pt \
      --joiner-model-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace.pt \
      /path/to/foo.wav \
      /path/to/bar.wav \
      /path/to/baz.wav

How to use the exported models
------------------------------

Please refer to
`<https://k2-fsa.github.io/sherpa/python/streaming_asr/lstm/index.html>`_
for its usage in `sherpa <https://k2-fsa.github.io/sherpa/python/streaming_asr/lstm/index.html>`_.
You can also find pretrained models there.
