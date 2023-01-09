Distillation with HuBERT
============================

This totorial shows you how to perform knowledge distillation in ``icefall`` 
with the `LibriSpeech <https://www.openslr.org/12>`_ dataset. The distillation method
used here is **M**ulti **V**ector **Q**uantization knowledge distillation (MVQ-KD). 
Please have a look at our paper `Predicting Multi-Codebook Vector Quantization Indexes for Knowledge Distillation <https://arxiv.org/abs/2211.00508>_`
for more details about MVQ-KD.

.. Note::

    This tutorial is based on recipe
     `pruned_transducer_stateless4 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless4>`_.
    Currently, we only implemented MVQ-KD in this recipe. However, MVQ-KD is theoretically applicable to all recipes
    with only minor changes. Feel free to try out MVQ-KD in different recipes. If you
    encounter any problems, please open a issue here `icefall <https://github.com/k2-fsa/icefall/issues>_`

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe.

Data preparation
----------------

We first prepare necessary training data for ``LibriSpeech``. 
This is the same as in `Pruned_transducer_statelessX <pruned_transducer_stateless.rst>`
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
  $ ./prepare.sh --stage 0 --stop-stage 0 # run only stage 0
  $ ./prepare.sh --stage 2 --stop-stage 5 # run from stage 2 to stage 5

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

Here, we prepare necessary data for MVQ-KD. When performing MVQ