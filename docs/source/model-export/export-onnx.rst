Export to ONNX
==============

In this section, we describe how to export models to `ONNX`_.

In each recipe, there is a file called ``export-onnx.py``, which is used
to export trained models to `ONNX`_.

There is also a file named ``onnx_pretrained.py``, which you can use
the exported `ONNX`_ model in Python with `onnxruntime`_ to decode sound files.

sherpa-onnx
-----------

We have a separate repository `sherpa-onnx`_ for deploying your exported models
on various platforms such as:

  - iOS
  - Android
  - Raspberry Pi
  - Linux/macOS/Windows


Please see the documentation of `sherpa-onnx`_ for details:

  `<https://k2-fsa.github.io/sherpa/onnx/index.html>`_

Example
-------

In the following, we demonstrate how to export a streaming Zipformer pre-trained
model from
`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11>`_
to `ONNX`_.

Download the pre-trained model
------------------------------

.. hint::

   We assume you have installed `git-lfs`_.

.. code-block:: bash


  cd egs/librispeech/ASR

  repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  pushd $repo
  git lfs pull --include "data/lang_bpe_500/bpe.model"
  git lfs pull --include "exp/pretrained.pt"
  cd exp
  ln -s pretrained.pt epoch-99.pt
  popd

Export the model to ONNX
------------------------

.. code-block:: bash

  ./pruned_transducer_stateless7_streaming/export-onnx.py \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --decode-chunk-len 32 \
    --exp-dir $repo/exp/

.. warning::

   ``export-onnx.py`` from different recipes has different options.

   In the above example, ``--decode-chunk-len`` is specific for the
   streaming Zipformer. Other models won't have such an option.

It will generate the following 3 files in ``$repo/exp``

  - ``encoder-epoch-99-avg-1.onnx``
  - ``decoder-epoch-99-avg-1.onnx``
  - ``joiner-epoch-99-avg-1.onnx``

Decode sound files with exported ONNX models
--------------------------------------------

.. code-block:: bash

  ./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
    --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav
