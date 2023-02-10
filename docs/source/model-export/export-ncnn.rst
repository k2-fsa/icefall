Export to ncnn
==============

We support exporting both
`LSTM transducer models <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2>`_
and
`ConvEmformer transducer models <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_
to `ncnn <https://github.com/tencent/ncnn>`_.

We also provide `<https://github.com/k2-fsa/sherpa-ncnn>`_
performing speech recognition using ``ncnn`` with exported models.
It has been tested on Linux, macOS, Windows, ``Android``, and ``Raspberry Pi``.

`sherpa-ncnn`_ is self-contained and can be statically linked to produce
a binary containing everything needed. Please refer
to its documentation for details:

 - `<https://k2-fsa.github.io/sherpa/ncnn/index.html>`_


Export LSTM transducer models
-----------------------------

Please refer to :ref:`export-lstm-transducer-model-for-ncnn` for details.



Export ConvEmformer transducer models
-------------------------------------

We use the pre-trained model from the following repository as an example:

  - `<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05>`_

We will show you step by step how to export it to `ncnn`_ and run it with `sherpa-ncnn`_.

.. hint::

  We use ``Ubuntu 18.04``, ``torch 1.10``, and ``Python 3.8`` for testing.

.. caution::

  Please use a more recent version of PyTorch. For instance, ``torch 1.8``
  may ``not`` work.

1. Download the pre-trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

  You can also refer to `<https://k2-fsa.github.io/sherpa/cpp/pretrained_models/online_transducer.html#icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05>`_ to download the pre-trained model.

  You have to install `git-lfs`_ before you continue.

.. code-block:: bash

  cd egs/librispeech/ASR

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

  git lfs pull --include "exp/pretrained-epoch-30-avg-10-averaged.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"

  cd ..

.. note::

  We download ``exp/pretrained-xxx.pt``, not ``exp/cpu-jit_xxx.pt``.


In the above code, we download the pre-trained model into the directory
``egs/librispeech/ASR/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05``.

2. Install ncnn and pnnx
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  # We put ncnn into $HOME/open-source/ncnn
  # You can change it to anywhere you like

  cd $HOME
  mkdir -p open-source
  cd open-source

  git clone https://github.com/csukuangfj/ncnn
  cd ncnn
  git submodule update --recursive --init

  # Note: We don't use "python setup.py install" or "pip install ." here

  mkdir -p build-wheel
  cd build-wheel

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_PYTHON=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TOOLS=ON \
  ..

  make -j4

  cd ..

  # Note: $PWD here is $HOME/open-source/ncnn

  export PYTHONPATH=$PWD/python:$PYTHONPATH
  export PATH=$PWD/tools/pnnx/build/src:$PATH
  export PATH=$PWD/build-wheel/tools/quantize:$PATH

  # Now build pnnx
  cd tools/pnnx
  mkdir build
  cd build
  cmake ..
  make -j4

  ./src/pnnx

Congratulations! You have successfully installed the following components:

  - ``pnxx``, which is an executable located in
    ``$HOME/open-source/ncnn/tools/pnnx/build/src``. We will use
    it to convert models exported by ``torch.jit.trace()``.
  - ``ncnn2int8``, which is an executable located in
    ``$HOME/open-source/ncnn/build-wheel/tools/quantize``. We will use
    it to quantize our models to ``int8``.
  - ``ncnn.cpython-38-x86_64-linux-gnu.so``, which is a Python module located
    in ``$HOME/open-source/ncnn/python/ncnn``.

    .. note::

      I am using ``Python 3.8``, so it
      is ``ncnn.cpython-38-x86_64-linux-gnu.so``. If you use a different
      version, say, ``Python 3.9``, the name would be
      ``ncnn.cpython-39-x86_64-linux-gnu.so``.

      Also, if you are not using Linux, the file name would also be different.
      But that does not matter. As long as you can compile it, it should work.

We have set up ``PYTHONPATH`` so that you can use ``import ncnn`` in your
Python code. We have also set up ``PATH`` so that you can use
``pnnx`` and ``ncnn2int8`` later in your terminal.

.. caution::

  Please don't use `<https://github.com/tencent/ncnn>`_.
  We have made some modifications to the offical `ncnn`_.

  We will synchronize `<https://github.com/csukuangfj/ncnn>`_ periodically
  with the official one.

3. Export the model via torch.jit.trace()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let us rename our pre-trained model:

.. code-block::

  cd egs/librispeech/ASR

  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp

  ln -s pretrained-epoch-30-avg-10-averaged.pt epoch-30.pt

  cd ../..

Next, we use the following code to export our model:

.. code-block:: bash

  dir=./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/

  ./conv_emformer_transducer_stateless2/export-for-ncnn.py \
    --exp-dir $dir/exp \
    --bpe-model $dir/data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 1 \
    --use-averaged-model 0 \
    \
    --num-encoder-layers 12 \
    --chunk-length 32 \
    --cnn-module-kernel 31 \
    --left-context-length 32 \
    --right-context-length 8 \
    --memory-size 32 \
    --encoder-dim 512

.. hint::

  We have renamed our model to ``epoch-30.pt`` so that we can use ``--epoch 30``.
  There is only one pre-trained model, so we use ``--avg 1 --use-averaged-model 0``.

  If you have trained a model by yourself and if you have all checkpoints
  available, please first use ``decode.py`` to tune ``--epoch --avg``
  and select the best combination with with ``--use-averaged-model 1``.

.. note::

  You will see the following log output:

  .. literalinclude:: ./code/export-conv-emformer-transducer-for-ncnn-output.txt

  The log shows the model has ``75490012`` parameters, i.e., ``~75 M``.

  .. code-block::

    ls -lh icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/pretrained-epoch-30-avg-10-averaged.pt

    -rw-r--r-- 1 kuangfangjun root 289M Jan 11 12:05 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/pretrained-epoch-30-avg-10-averaged.pt

  You can see that the file size of the pre-trained model is ``289 MB``, which
  is roughly ``75490012*4/1024/1024 = 287.97 MB``.

After running ``conv_emformer_transducer_stateless2/export-for-ncnn.py``,
we will get the following files:

.. code-block:: bash

  ls -lh icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/*pnnx*

  -rw-r--r-- 1 kuangfangjun root 1010K Jan 11 12:15 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.pt
  -rw-r--r-- 1 kuangfangjun root  283M Jan 11 12:15 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.pt
  -rw-r--r-- 1 kuangfangjun root  3.0M Jan 11 12:15 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.pt


.. _conv-emformer-step-3-export-torchscript-model-via-pnnx:

3. Export torchscript model via pnnx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

  Make sure you have set up the ``PATH`` environment variable. Otherwise,
  it will throw an error saying that ``pnnx`` could not be found.

Now, it's time to export our models to `ncnn`_ via ``pnnx``.

.. code-block::

  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

  pnnx ./encoder_jit_trace-pnnx.pt
  pnnx ./decoder_jit_trace-pnnx.pt
  pnnx ./joiner_jit_trace-pnnx.pt

It will generate the following files:

.. code-block:: bash

  ls -lh  icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/*ncnn*{bin,param}

  -rw-r--r-- 1 kuangfangjun root 503K Jan 11 12:38 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  437 Jan 11 12:38 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 142M Jan 11 12:36 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  79K Jan 11 12:36 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 1.5M Jan 11 12:38 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  488 Jan 11 12:38 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.ncnn.param

There are two types of files:

- ``param``: It is a text file containing the model architectures. You can
  use a text editor to view its content.
- ``bin``: It is a binary file containing the model parameters.

We compare the file sizes of the models below before and after converting via ``pnnx``:

.. see https://tableconvert.com/restructuredtext-generator

+----------------------------------+------------+
| File name                        | File size  |
+==================================+============+
| encoder_jit_trace-pnnx.pt        | 283 MB     |
+----------------------------------+------------+
| decoder_jit_trace-pnnx.pt        | 1010 KB    |
+----------------------------------+------------+
| joiner_jit_trace-pnnx.pt         | 3.0 MB     |
+----------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin  | 142 MB     |
+----------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin  | 503 KB     |
+----------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin   | 1.5 MB     |
+----------------------------------+------------+

You can see that the file sizes of the models after conversion are about one half
of the models before conversion:

  - encoder: 283 MB vs 142 MB
  - decoder: 1010 KB vs 503 KB
  - joiner: 3.0 MB vs 1.5 MB

The reason is that by default ``pnnx`` converts ``float32`` parameters
to ``float16``. A ``float32`` parameter occupies 4 bytes, while it is 2 bytes
for ``float16``. Thus, it is ``twice smaller`` after conversion.

.. hint::

  If you use ``pnnx ./encoder_jit_trace-pnnx.pt fp16=0``, then ``pnnx``
  won't convert ``float32`` to ``float16``.

4. Test the exported models in icefall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  We assume you have set up the environment variable ``PYTHONPATH`` when
  building `ncnn`_.

Now we have successfully converted our pre-trained model to `ncnn`_ format.
The generated 6 files are what we need. You can use the following code to
test the converted models:

.. code-block:: bash

  ./conv_emformer_transducer_stateless2/streaming-ncnn-decode.py \
    --tokens ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/tokens.txt \
    --encoder-param-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.ncnn.param \
    --encoder-bin-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.ncnn.bin \
    --decoder-param-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.ncnn.param \
    --decoder-bin-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.ncnn.bin \
    --joiner-param-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.ncnn.param \
    --joiner-bin-filename ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.ncnn.bin \
    ./icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/test_wavs/1089-134686-0001.wav

.. hint::

  `ncnn`_ supports only ``batch size == 1``, so ``streaming-ncnn-decode.py`` accepts
  only 1 wave file as input.

The output is given below:

.. literalinclude:: ./code/test-stremaing-ncnn-decode-conv-emformer-transducer-libri.txt

Congratulations! You have successfully exported a model from PyTorch to `ncnn`_!


.. _conv-emformer-modify-the-exported-encoder-for-sherpa-ncnn:

5. Modify the exported encoder for sherpa-ncnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use the exported models in `sherpa-ncnn`_, we have to modify
``encoder_jit_trace-pnnx.ncnn.param``.

Let us have a look at the first few lines of ``encoder_jit_trace-pnnx.ncnn.param``:

.. code-block::

  7767517
  1060 1342
  Input                    in0                      0 1 in0

**Explanation** of the above three lines:

  1. ``7767517``, it is a magic number and should not be changed.
  2. ``1060 1342``, the first number ``1060`` specifies the number of layers
     in this file, while ``1342`` specifies the number of intermediate outputs
     of this file
  3. ``Input in0 0 1 in0``, ``Input`` is the layer type of this layer; ``in0``
     is the layer name of this layer; ``0`` means this layer has no input;
     ``1`` means this layer has one output; ``in0`` is the output name of
     this layer.

We need to add 1 extra line and also increment the number of layers.
The result looks like below:

.. code-block:: bash

  7767517
  1061 1342
  SherpaMetaData           sherpa_meta_data1        0 0 0=1 1=12 2=32 3=31 4=8 5=32 6=8 7=512
  Input                    in0                      0 1 in0

**Explanation**

  1. ``7767517``, it is still the same
  2. ``1061 1342``, we have added an extra layer, so we need to update ``1060`` to ``1061``.
     We don't need to change ``1342`` since the newly added layer has no inputs or outputs.
  3. ``SherpaMetaData  sherpa_meta_data1  0 0 0=1 1=12 2=32 3=31 4=8 5=32 6=8 7=512``
     This line is newly added. Its explanation is given below:

      - ``SherpaMetaData`` is the type of this layer. Must be ``SherpaMetaData``.
      - ``sherpa_meta_data1`` is the name of this layer. Must be ``sherpa_meta_data1``.
      - ``0 0`` means this layer has no inputs or output. Must be ``0 0``
      - ``0=1``, 0 is the key and 1 is the value. MUST be ``0=1``
      - ``1=12``, 1 is the key and 12 is the value of the
        parameter ``--num-encoder-layers`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.
      - ``2=32``, 2 is the key and 32 is the value of the
        parameter ``--memory-size`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.
      - ``3=31``, 3 is the key and 31 is the value of the
        parameter ``--cnn-module-kernel`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.
      - ``4=8``, 4 is the key and 8 is the value of the
        parameter ``--left-context-length`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.
      - ``5=32``, 5 is the key and 32 is the value of the
        parameter ``--chunk-length`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.
      - ``6=8``, 6 is the key and 8 is the value of the
        parameter ``--right-context-length`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.
      - ``7=512``, 7 is the key and 512 is the value of the
        parameter ``--encoder-dim`` that you provided when running
        ``conv_emformer_transducer_stateless2/export-for-ncnn.py``.

      For ease of reference, we list the key-value pairs that you need to add
      in the following table. If your model has a different setting, please
      change the values for ``SherpaMetaData`` accordingly. Otherwise, you
      will be ``SAD``.

          +------+-----------------------------+
          | key  | value                       |
          +======+=============================+
          | 0    | 1 (fixed)                   |
          +------+-----------------------------+
          | 1    | ``--num-encoder-layers``    |
          +------+-----------------------------+
          | 2    | ``--memory-size``           |
          +------+-----------------------------+
          | 3    | ``--cnn-module-kernel``     |
          +------+-----------------------------+
          | 4    | ``--left-context-length``   |
          +------+-----------------------------+
          | 5    | ``--chunk-length``          |
          +------+-----------------------------+
          | 6    | ``--right-context-length``  |
          +------+-----------------------------+
          | 7    | ``--encoder-dim``           |
          +------+-----------------------------+

  4. ``Input in0 0 1 in0``. No need to change it.

.. caution::

  When you add a new layer ``SherpaMetaData``, please remember to update the
  number of layers. In our case, update  ``1060`` to ``1061``. Otherwise,
  you will be SAD later.

.. hint::

  After adding the new layer ``SherpaMetaData``, you cannot use this model
  with ``streaming-ncnn-decode.py`` anymore since ``SherpaMetaData`` is
  supported only in `sherpa-ncnn`_.

.. hint::

  `ncnn`_ is very flexible. You can add new layers to it just by text-editing
  the ``param`` file! You don't need to change the ``bin`` file.

Now you can use this model in `sherpa-ncnn`_.
Please refer to the following documentation:

  - Linux/macOS/Windows/arm/aarch64: `<https://k2-fsa.github.io/sherpa/ncnn/install/index.html>`_
  - Android: `<https://k2-fsa.github.io/sherpa/ncnn/android/index.html>`_
  - Python: `<https://k2-fsa.github.io/sherpa/ncnn/python/index.html>`_

We have a list of pre-trained models that have been exported for `sherpa-ncnn`_:

  - `<https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html>`_

    You can find more usages there.

6. (Optional) int8 quantization with sherpa-ncnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is optional.

In this step, we describe how to quantize our model with ``int8``.

Change :ref:`conv-emformer-step-3-export-torchscript-model-via-pnnx` to
disable ``fp16`` when using ``pnnx``:

.. code-block::

  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

  pnnx ./encoder_jit_trace-pnnx.pt fp16=0
  pnnx ./decoder_jit_trace-pnnx.pt
  pnnx ./joiner_jit_trace-pnnx.pt fp16=0

.. note::

  We add ``fp16=0`` when exporting the encoder and joiner. `ncnn`_ does not
  support quantizing the decoder model yet. We will update this documentation
  once `ncnn`_ supports it. (Maybe in this year, 2023).

It will generate the following files

.. code-block:: bash

  ls -lh icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/*_jit_trace-pnnx.ncnn.{param,bin}

  -rw-r--r-- 1 kuangfangjun root 503K Jan 11 15:56 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  437 Jan 11 15:56 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/decoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 283M Jan 11 15:56 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  79K Jan 11 15:56 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/encoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 3.0M Jan 11 15:56 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  488 Jan 11 15:56 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/joiner_jit_trace-pnnx.ncnn.param

Let us compare again the file sizes:

+----------------------------------------+------------+
| File name                              | File size  |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.pt              | 283 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.pt              | 1010 KB    |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.pt               | 3.0 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp16) | 142 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin (fp16) | 503 KB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp16) | 1.5 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp32) | 283 MB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp32) | 3.0 MB     |
+----------------------------------------+------------+

You can see that the file sizes are doubled when we disable ``fp16``.

.. note::

  You can again use ``streaming-ncnn-decode.py`` to test the exported models.

Next, follow :ref:`conv-emformer-modify-the-exported-encoder-for-sherpa-ncnn`
to modify ``encoder_jit_trace-pnnx.ncnn.param``.

Change

.. code-block:: bash

  7767517
  1060 1342
  Input                    in0                      0 1 in0

to

.. code-block:: bash

  7767517
  1061 1342
  SherpaMetaData           sherpa_meta_data1        0 0 0=1 1=12 2=32 3=31 4=8 5=32 6=8 7=512
  Input                    in0                      0 1 in0

.. caution::

  Please follow :ref:`conv-emformer-modify-the-exported-encoder-for-sherpa-ncnn`
  to change the values for ``SherpaMetaData`` if your model uses a different setting.


Next, let us compile `sherpa-ncnn`_ since we will quantize our models within
`sherpa-ncnn`_.

.. code-block:: bash

  # We will download sherpa-ncnn to $HOME/open-source/
  # You can change it to anywhere you like.
  cd $HOME
  mkdir -p open-source

  cd open-source
  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  mkdir build
  cd build
  cmake ..
  make -j 4

  ./bin/generate-int8-scale-table

  export PATH=$HOME/open-source/sherpa-ncnn/build/bin:$PATH

The output of the above commands are:

.. code-block:: bash

  (py38) kuangfangjun:build$ generate-int8-scale-table
  Please provide 10 arg. Currently given: 1
  Usage:
  generate-int8-scale-table encoder.param encoder.bin decoder.param decoder.bin joiner.param joiner.bin encoder-scale-table.txt joiner-scale-table.txt wave_filenames.txt

  Each line in wave_filenames.txt is a path to some 16k Hz mono wave file.

We need to create a file ``wave_filenames.txt``, in which we need to put
some calibration wave files. For testing purpose, we put the ``test_wavs``
from the pre-trained model repository `<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05>`_

.. code-block:: bash

  cd egs/librispeech/ASR
  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

  cat <<EOF > wave_filenames.txt
  ../test_wavs/1089-134686-0001.wav
  ../test_wavs/1221-135766-0001.wav
  ../test_wavs/1221-135766-0002.wav
  EOF

Now we can calculate the scales needed for quantization with the calibration data:

.. code-block:: bash

  cd egs/librispeech/ASR
  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

  generate-int8-scale-table \
    ./encoder_jit_trace-pnnx.ncnn.param \
    ./encoder_jit_trace-pnnx.ncnn.bin \
    ./decoder_jit_trace-pnnx.ncnn.param \
    ./decoder_jit_trace-pnnx.ncnn.bin \
    ./joiner_jit_trace-pnnx.ncnn.param \
    ./joiner_jit_trace-pnnx.ncnn.bin \
    ./encoder-scale-table.txt \
    ./joiner-scale-table.txt \
    ./wave_filenames.txt

The output logs are in the following:

.. literalinclude:: ./code/generate-int-8-scale-table-for-conv-emformer.txt

It generates the following two files:

.. code-block:: bash

  $ ls -lh encoder-scale-table.txt joiner-scale-table.txt
  -rw-r--r-- 1 kuangfangjun root 955K Jan 11 17:28 encoder-scale-table.txt
  -rw-r--r-- 1 kuangfangjun root  18K Jan 11 17:28 joiner-scale-table.txt

.. caution::

  Definitely, you need more calibration data to compute the scale table.

Finally, let us use the scale table to quantize our models into ``int8``.

.. code-block:: bash

  ncnn2int8

  usage: ncnn2int8 [inparam] [inbin] [outparam] [outbin] [calibration table]

First, we quantize the encoder model:

.. code-block:: bash

  cd egs/librispeech/ASR
  cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

  ncnn2int8 \
    ./encoder_jit_trace-pnnx.ncnn.param \
    ./encoder_jit_trace-pnnx.ncnn.bin \
    ./encoder_jit_trace-pnnx.ncnn.int8.param \
    ./encoder_jit_trace-pnnx.ncnn.int8.bin \
    ./encoder-scale-table.txt

Next, we quantize the joiner model:

.. code-block:: bash

  ncnn2int8 \
    ./joiner_jit_trace-pnnx.ncnn.param \
    ./joiner_jit_trace-pnnx.ncnn.bin \
    ./joiner_jit_trace-pnnx.ncnn.int8.param \
    ./joiner_jit_trace-pnnx.ncnn.int8.bin \
    ./joiner-scale-table.txt

The above two commands generate the following 4 files:

.. code-block:: bash

  -rw-r--r-- 1 kuangfangjun root  99M Jan 11 17:34 encoder_jit_trace-pnnx.ncnn.int8.bin
  -rw-r--r-- 1 kuangfangjun root  78K Jan 11 17:34 encoder_jit_trace-pnnx.ncnn.int8.param
  -rw-r--r-- 1 kuangfangjun root 774K Jan 11 17:35 joiner_jit_trace-pnnx.ncnn.int8.bin
  -rw-r--r-- 1 kuangfangjun root  496 Jan 11 17:35 joiner_jit_trace-pnnx.ncnn.int8.param

Congratulations! You have successfully quantized your model from ``float32`` to ``int8``.

.. caution::

  ``ncnn.int8.param`` and ``ncnn.int8.bin`` must be used in pairs.

  You can replace ``ncnn.param`` and ``ncnn.bin`` with ``ncnn.int8.param``
  and ``ncnn.int8.bin`` in `sherpa-ncnn`_ if you like.

  For instance, to use only the ``int8`` encoder in ``sherpa-ncnn``, you can
  replace the following invocation:

    .. code-block::

      cd egs/librispeech/ASR
      cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

      sherpa-ncnn \
        ../data/lang_bpe_500/tokens.txt \
        ./encoder_jit_trace-pnnx.ncnn.param \
        ./encoder_jit_trace-pnnx.ncnn.bin \
        ./decoder_jit_trace-pnnx.ncnn.param \
        ./decoder_jit_trace-pnnx.ncnn.bin \
        ./joiner_jit_trace-pnnx.ncnn.param \
        ./joiner_jit_trace-pnnx.ncnn.bin \
        ../test_wavs/1089-134686-0001.wav

  with

    .. code-block::

      cd egs/librispeech/ASR
      cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/

      sherpa-ncnn \
        ../data/lang_bpe_500/tokens.txt \
        ./encoder_jit_trace-pnnx.ncnn.int8.param \
        ./encoder_jit_trace-pnnx.ncnn.int8.bin \
        ./decoder_jit_trace-pnnx.ncnn.param \
        ./decoder_jit_trace-pnnx.ncnn.bin \
        ./joiner_jit_trace-pnnx.ncnn.param \
        ./joiner_jit_trace-pnnx.ncnn.bin \
        ../test_wavs/1089-134686-0001.wav


The following table compares again the file sizes:


+----------------------------------------+------------+
| File name                              | File size  |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.pt              | 283 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.pt              | 1010 KB    |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.pt               | 3.0 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp16) | 142 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin (fp16) | 503 KB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp16) | 1.5 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp32) | 283 MB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp32) | 3.0 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.int8.bin   | 99 MB      |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.int8.bin    | 774 KB     |
+----------------------------------------+------------+

You can see that the file sizes of the model after ``int8`` quantization
are much smaller.

.. hint::

    Currently, only linear layers and convolutional layers are quantized
    with ``int8``, so you don't see an exact ``4x`` reduction in file sizes.

.. note::

  You need to test the recognition accuracy after ``int8`` quantization.

You can find the speed comparison at `<https://github.com/k2-fsa/sherpa-ncnn/issues/44>`_.


That's it! Have fun with `sherpa-ncnn`_!
