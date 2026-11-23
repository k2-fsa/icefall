Zipformer faster training with fused GPU kernels
================================================

This page shows how to make Zipformer training faster, i.e., use less GPU
time, with the optional fused GPU kernels in
`zipformer/fast_kernels.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/fast_kernels.py>`_.

A Zipformer training step is limited by memory bandwidth: most of the GPU
time goes to elementwise and copy kernels, not to matrix multiplications.
``fast_kernels.py`` replaces the most memory-bound parts of
`zipformer/scaling.py <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/scaling.py>`_
with fused `Triton <https://github.com/triton-lang/triton>`_ kernels and
cheaper backward functions. They compute the same values as the reference
code (up to floating-point rounding) and only change training speed.
Decoding, model export and the saved checkpoints are not affected.

.. note::

   The optimizations are **off by default**. Without the environment variable
   below, training behaves exactly as before.

Enabling the optimizations
--------------------------

Set the environment variable ``ZIPFORMER_FAST_KERNELS``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Effect
   * - unset, ``0``
     - upstream behaviour (default)
   * - ``1`` or ``all``
     - enable all features listed below
   * - comma-separated list, e.g., ``balancer,softmax``
     - enable only the listed features

The features are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Feature
     - What it replaces
   * - ``actlin``
     - ``ActivationDropoutAndLinear``: one Triton kernel computes
       ``swoosh(x) * dropout_mask`` and writes the matmul dtype directly;
       fused backward
   * - ``swoosh``
     - ``SwooshL`` / ``SwooshR`` modules: a Triton kernel that keeps the input
       dtype (``k2.swoosh_l`` / ``k2.swoosh_r`` return float32)
   * - ``softmax``
     - ``SoftmaxFunction`` backward: PyTorch's fused softmax backward instead of
       float32 elementwise operations
   * - ``balancer``
     - ``BalancerFunction`` backward: closed-form gradient (two small kernels)
       instead of re-running a float32 autograd graph
   * - ``channels_last``
     - ``Conv2dSubsampling`` runs in NHWC layout so that cuDNN's fast depthwise
       convolution kernels are used

.. hint::

   The optimizations need a CUDA GPU and Triton, which is installed together
   with PyTorch on Linux. If either is missing, a warning is printed and the
   reference code is used.

Training
--------

Put the environment variable in front of the usual training command. The
following example trains a Zipformer (medium) model with all optimizations on
4 GPUs:

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ export CUDA_VISIBLE_DEVICES="0,1,2,3"
  $ ZIPFORMER_FAST_KERNELS=1 ./zipformer/train.py \
      --world-size 4 \
      --num-epochs 30 \
      --start-epoch 1 \
      --use-fp16 1 \
      --exp-dir zipformer/exp \
      --max-duration 1000

It works the same way for the other model sizes, for example the large model:

.. code-block:: bash

  $ ZIPFORMER_FAST_KERNELS=1 ./zipformer/train.py \
      --world-size 4 \
      --num-epochs 30 \
      --start-epoch 1 \
      --use-fp16 1 \
      --exp-dir zipformer/exp-large \
      --num-encoder-layers 2,2,4,5,4,2 \
      --feedforward-dim 512,768,1536,2048,1536,768 \
      --encoder-dim 192,256,512,768,512,256 \
      --encoder-unmasked-dim 192,192,256,320,256,192 \
      --max-duration 1000

To enable only some of the optimizations:

.. code-block:: bash

  $ ZIPFORMER_FAST_KERNELS=balancer,softmax ./zipformer/train.py ...

Decoding
--------

Decoding needs no change; do not set the environment variable:

.. code-block:: bash

  $ ./zipformer/decode.py \
      --epoch 30 \
      --avg 9 \
      --exp-dir zipformer/exp \
      --max-duration 600 \
      --decoding-method greedy_search

For a model trained with non-default architecture options, pass the same
options as in training, e.g., for the large model above:

.. code-block:: bash

  $ ./zipformer/decode.py \
      --epoch 30 \
      --avg 9 \
      --exp-dir zipformer/exp-large \
      --num-encoder-layers 2,2,4,5,4,2 \
      --feedforward-dim 512,768,1536,2048,1536,768 \
      --encoder-dim 192,256,512,768,512,256 \
      --encoder-unmasked-dim 192,192,256,320,256,192 \
      --max-duration 600 \
      --decoding-method greedy_search

Testing the kernels
-------------------

``zipformer/test_fast_kernels.py`` compares every replacement against the
reference implementation in ``scaling.py`` in float32 and bfloat16
(a GPU is required):

.. code-block:: bash

  $ cd egs/librispeech/ASR
  $ python3 ./zipformer/test_fast_kernels.py

Results
-------

Zipformer medium (65.5 M parameters), LibriSpeech ``train-clean-100``, fp16,
``--max-duration 1000``, one NVIDIA L40S GPU per run, 30 epochs, same seed:

.. list-table::
   :header-rows: 1

   * -
     - upstream
     - ``ZIPFORMER_FAST_KERNELS=1``
   * - total training time
     - 7503 s
     - 5641 s (1.33x faster, 25% less GPU time)
   * - WER (test-clean / test-other), greedy search
     - 7.53 / 20.20
     - 7.43 / 20.31
   * - WER (test-clean / test-other), modified beam search
     - 7.39 / 19.62
     - 7.27 / 19.73

For a single training step (bf16, ``--max-duration 1000``), throughput goes
from 1577 to 2161 seconds of audio per second (+36%).

Zipformer large (148.4 M parameters, the architecture options shown in
`Training`_), same data and settings, 4 epochs, the two runs at the same time
on two NVIDIA L40S GPUs:

.. list-table::
   :header-rows: 1

   * -
     - upstream
     - ``ZIPFORMER_FAST_KERNELS=1``
   * - total training time (4 epochs)
     - 1494 s
     - 1017 s (1.47x faster, 32% less GPU time)
   * - validation loss after epoch 3
     - 0.796
     - 0.642
   * - peak GPU memory
     - 20.6 GB
     - 21.3 GB

This run measures speed only: 4 epochs are too few to compare the final WER,
and the validation losses differ by run-to-run noise this early in training.

For a single training step of the large model (``--max-duration 1000``),
throughput goes from 1265 to 1675 seconds of audio per second with bf16
(+32%) and from 1275 to 1765 with fp16 (+38%).

.. note::

   Dropout random numbers are drawn differently, so the losses are not
   bit-identical to upstream. The WER is the same within run-to-run noise.
