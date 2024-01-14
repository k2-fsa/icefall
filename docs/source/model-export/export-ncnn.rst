.. _icefall_export_to_ncnn:

Export to ncnn
==============

We support exporting the following models
to `ncnn <https://github.com/tencent/ncnn>`_:

  - `Zipformer transducer models <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_

  - `LSTM transducer models <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2>`_

  - `ConvEmformer transducer models <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_

We also provide `sherpa-ncnn`_
for performing speech recognition using `ncnn`_ with exported models.
It has been tested on the following platforms:

  - Linux
  - macOS
  - Windows
  - ``Android``
  - ``iOS``
  - ``Raspberry Pi``
  - `爱芯派 <https://wiki.sipeed.com/hardware/zh/>`_ (`MAIX-III AXera-Pi <https://wiki.sipeed.com/hardware/en/maixIII/ax-pi/axpi.html>`_).
  - `RV1126 <https://www.rock-chips.com/a/en/products/RV11_Series/2020/0427/1076.html>`_

`sherpa-ncnn`_ is self-contained and can be statically linked to produce
a binary containing everything needed. Please refer
to its documentation for details:

 - `<https://k2-fsa.github.io/sherpa/ncnn/index.html>`_


.. toctree::

   export-ncnn-zipformer
   export-ncnn-conv-emformer
   export-ncnn-lstm
