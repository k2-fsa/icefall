Export to ncnn
==============

We support exporting LSTM transducer models to `ncnn <https://github.com/tencent/ncnn>`_.

Please refer to :ref:`export-model-for-ncnn` for details.

We also provide `<https://github.com/k2-fsa/sherpa-ncnn>`_
performing speech recognition using ``ncnn`` with exported models.
It has been tested on Linux, macOS, Windows, and Raspberry Pi. The project is
self-contained and can be statically linked to produce a binary containing
everything needed.
