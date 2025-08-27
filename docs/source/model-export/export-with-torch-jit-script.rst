.. _export-model-with-torch-jit-script:

Export model with torch.jit.script()
====================================

In this section, we describe how to export a model via
``torch.jit.script()``.

When to use it
--------------

If we want to use our trained model with torchscript,
we can use ``torch.jit.script()``.

.. hint::

  See :ref:`export-model-with-torch-jit-trace`
  if you want to use ``torch.jit.trace()``.

How to export
-------------

We use
`<https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless3>`_
as an example in the following.

.. code-block:: bash

    cd egs/librispeech/ASR
    epoch=14
    avg=1

    ./pruned_transducer_stateless3/export.py \
      --exp-dir ./pruned_transducer_stateless3/exp \
      --tokens data/lang_bpe_500/tokens.txt \
      --epoch $epoch \
      --avg $avg \
      --jit 1

It will generate a file ``cpu_jit.pt`` in ``pruned_transducer_stateless3/exp``.

.. caution::

   Don't be confused by ``cpu`` in ``cpu_jit.pt``. We move all parameters
   to CPU before saving it into a ``pt`` file; that's why we use ``cpu``
   in the filename.

How to use the exported model
-----------------------------

Please refer to the following pages for usage:

- `<https://k2-fsa.github.io/sherpa/python/streaming_asr/emformer/index.html>`_
- `<https://k2-fsa.github.io/sherpa/python/streaming_asr/conv_emformer/index.html>`_
- `<https://k2-fsa.github.io/sherpa/python/streaming_asr/conformer/index.html>`_
- `<https://k2-fsa.github.io/sherpa/python/offline_asr/conformer/index.html>`_
- `<https://k2-fsa.github.io/sherpa/cpp/offline_asr/gigaspeech.html>`_
- `<https://k2-fsa.github.io/sherpa/cpp/offline_asr/wenetspeech.html>`_
