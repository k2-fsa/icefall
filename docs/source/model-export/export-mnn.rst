.. _icefall_export_to_mnn:

Export to mnn
==============

We support exporting the following models
to `mnn <https://github.com/alibaba/MNN>`_:

  - `Zipformer transducer models <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_

We also provide `sherpa-mnn`_
for performing speech recognition using `MNN`_ with exported models.
It has been tested on the following platforms:

  - Linux
  - RK3588s

`sherpa-mnn`_ is self-contained and can be statically linked to produce
a binary containing everything needed. Please refer
to its documentation for details:

 - `<https://k2-fsa.github.io/sherpa/mnn/index.html>`_


.. toctree::

   export-mnn-zipformer
