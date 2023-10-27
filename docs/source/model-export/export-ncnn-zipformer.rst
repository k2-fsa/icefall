.. _export_streaming_zipformer_transducer_models_to_ncnn:

Export streaming Zipformer transducer models to ncnn
----------------------------------------------------

We use the pre-trained model from the following repository as an example:

`<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29>`_

We will show you step by step how to export it to `ncnn`_ and run it with `sherpa-ncnn`_.

.. hint::

  We use ``Ubuntu 18.04``, ``torch 1.13``, and ``Python 3.8`` for testing.

.. caution::

  Please use a more recent version of PyTorch. For instance, ``torch 1.8``
  may ``not`` work.

1. Download the pre-trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

  You have to install `git-lfs`_ before you continue.


.. code-block:: bash

  cd egs/librispeech/ASR
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
  cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

  git lfs pull --include "exp/pretrained.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"

  cd ..

.. note::

  We downloaded ``exp/pretrained-xxx.pt``, not ``exp/cpu-jit_xxx.pt``.

In the above code, we downloaded the pre-trained model into the directory
``egs/librispeech/ASR/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29``.

2. Install ncnn and pnnx
^^^^^^^^^^^^^^^^^^^^^^^^

Please refer to :ref:`export_for_ncnn_install_ncnn_and_pnnx` .


3. Export the model via torch.jit.trace()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let us rename our pre-trained model:

.. code-block::

  cd egs/librispeech/ASR

  cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp

  ln -s pretrained.pt epoch-99.pt

  cd ../..

Next, we use the following code to export our model:

.. code-block:: bash

  dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

  ./pruned_transducer_stateless7_streaming/export-for-ncnn.py \
    --tokens $dir/data/lang_bpe_500/tokens.txt \
    --exp-dir $dir/exp \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --decode-chunk-len 32 \
    --num-left-chunks 4 \
    --num-encoder-layers "2,4,3,2,4" \
    --feedforward-dims "1024,1024,2048,2048,1024" \
    --nhead "8,8,8,8,8" \
    --encoder-dims "384,384,384,384,384" \
    --attention-dims "192,192,192,192,192" \
    --encoder-unmasked-dims "256,256,256,256,256" \
    --zipformer-downsampling-factors "1,2,4,8,2" \
    --cnn-module-kernels "31,31,31,31,31" \
    --decoder-dim 512 \
    --joiner-dim 512

.. caution::

   If your model has different configuration parameters, please change them accordingly.

.. hint::

  We have renamed our model to ``epoch-99.pt`` so that we can use ``--epoch 99``.
  There is only one pre-trained model, so we use ``--avg 1 --use-averaged-model 0``.

  If you have trained a model by yourself and if you have all checkpoints
  available, please first use ``decode.py`` to tune ``--epoch --avg``
  and select the best combination with with ``--use-averaged-model 1``.

.. note::

  You will see the following log output:

  .. literalinclude:: ./code/export-zipformer-transducer-for-ncnn-output.txt

  The log shows the model has ``69920376`` parameters, i.e., ``~69.9 M``.

  .. code-block:: bash

   ls -lh icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/pretrained.pt
   -rw-r--r-- 1 kuangfangjun root 269M Jan 12 12:53 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/pretrained.pt

  You can see that the file size of the pre-trained model is ``269 MB``, which
  is roughly equal to ``69920376*4/1024/1024 = 266.725 MB``.

After running ``pruned_transducer_stateless7_streaming/export-for-ncnn.py``,
we will get the following files:

.. code-block:: bash

  ls -lh icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/*pnnx.pt

  -rw-r--r-- 1 kuangfangjun root 1022K Feb 27 20:23 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/decoder_jit_trace-pnnx.pt
  -rw-r--r-- 1 kuangfangjun root  266M Feb 27 20:23 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/encoder_jit_trace-pnnx.pt
  -rw-r--r-- 1 kuangfangjun root  2.8M Feb 27 20:23 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/joiner_jit_trace-pnnx.pt

.. _zipformer-transducer-step-4-export-torchscript-model-via-pnnx:

4. Export torchscript model via pnnx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

  Make sure you have set up the ``PATH`` environment variable
  in :ref:`export_for_ncnn_install_ncnn_and_pnnx`. Otherwise,
  it will throw an error saying that ``pnnx`` could not be found.

Now, it's time to export our models to `ncnn`_ via ``pnnx``.

.. code-block::

  cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/

  pnnx ./encoder_jit_trace-pnnx.pt
  pnnx ./decoder_jit_trace-pnnx.pt
  pnnx ./joiner_jit_trace-pnnx.pt

It will generate the following files:

.. code-block:: bash

  ls -lh  icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/*ncnn*{bin,param}

  -rw-r--r-- 1 kuangfangjun root 509K Feb 27 20:31 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  437 Feb 27 20:31 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/decoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 133M Feb 27 20:30 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root 152K Feb 27 20:30 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/encoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 1.4M Feb 27 20:31 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/joiner_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  488 Feb 27 20:31 icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/joiner_jit_trace-pnnx.ncnn.param

There are two types of files:

- ``param``: It is a text file containing the model architectures. You can
  use a text editor to view its content.
- ``bin``: It is a binary file containing the model parameters.

We compare the file sizes of the models below before and after converting via ``pnnx``:

.. see https://tableconvert.com/restructuredtext-generator

+----------------------------------+------------+
| File name                        | File size  |
+==================================+============+
| encoder_jit_trace-pnnx.pt        | 266 MB     |
+----------------------------------+------------+
| decoder_jit_trace-pnnx.pt        | 1022 KB    |
+----------------------------------+------------+
| joiner_jit_trace-pnnx.pt         | 2.8 MB     |
+----------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin  | 133 MB     |
+----------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin  | 509 KB     |
+----------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin   | 1.4 MB     |
+----------------------------------+------------+

You can see that the file sizes of the models after conversion are about one half
of the models before conversion:

  - encoder: 266 MB vs 133 MB
  - decoder: 1022 KB vs 509 KB
  - joiner: 2.8 MB vs 1.4 MB

The reason is that by default ``pnnx`` converts ``float32`` parameters
to ``float16``. A ``float32`` parameter occupies 4 bytes, while it is 2 bytes
for ``float16``. Thus, it is ``twice smaller`` after conversion.

.. hint::

  If you use ``pnnx ./encoder_jit_trace-pnnx.pt fp16=0``, then ``pnnx``
  won't convert ``float32`` to ``float16``.

5. Test the exported models in icefall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  We assume you have set up the environment variable ``PYTHONPATH`` when
  building `ncnn`_.

Now we have successfully converted our pre-trained model to `ncnn`_ format.
The generated 6 files are what we need. You can use the following code to
test the converted models:

.. code-block:: bash

  python3 ./pruned_transducer_stateless7_streaming/streaming-ncnn-decode.py \
    --tokens ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt \
    --encoder-param-filename ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/encoder_jit_trace-pnnx.ncnn.param \
    --encoder-bin-filename ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/encoder_jit_trace-pnnx.ncnn.bin \
    --decoder-param-filename ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/decoder_jit_trace-pnnx.ncnn.param \
    --decoder-bin-filename ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/decoder_jit_trace-pnnx.ncnn.bin \
    --joiner-param-filename ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/joiner_jit_trace-pnnx.ncnn.param \
    --joiner-bin-filename ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/joiner_jit_trace-pnnx.ncnn.bin \
    ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav

.. hint::

  `ncnn`_ supports only ``batch size == 1``, so ``streaming-ncnn-decode.py`` accepts
  only 1 wave file as input.

The output is given below:

.. literalinclude:: ./code/test-streaming-ncnn-decode-zipformer-transducer-libri.txt

Congratulations! You have successfully exported a model from PyTorch to `ncnn`_!

.. _zipformer-modify-the-exported-encoder-for-sherpa-ncnn:

6. Modify the exported encoder for sherpa-ncnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use the exported models in `sherpa-ncnn`_, we have to modify
``encoder_jit_trace-pnnx.ncnn.param``.

Let us have a look at the first few lines of ``encoder_jit_trace-pnnx.ncnn.param``:

.. code-block::

  7767517
  2028 2547
  Input                    in0                      0 1 in0

**Explanation** of the above three lines:

  1. ``7767517``, it is a magic number and should not be changed.
  2. ``2028 2547``, the first number ``2028`` specifies the number of layers
     in this file, while ``2547`` specifies the number of intermediate outputs
     of this file
  3. ``Input in0 0 1 in0``, ``Input`` is the layer type of this layer; ``in0``
     is the layer name of this layer; ``0`` means this layer has no input;
     ``1`` means this layer has one output; ``in0`` is the output name of
     this layer.

We need to add 1 extra line and also increment the number of layers.
The result looks like below:

.. code-block:: bash

  7767517
  2029 2547
  SherpaMetaData           sherpa_meta_data1        0 0 0=2 1=32 2=4 3=7 15=1 -23316=5,2,4,3,2,4 -23317=5,384,384,384,384,384 -23318=5,192,192,192,192,192 -23319=5,1,2,4,8,2 -23320=5,31,31,31,31,31
  Input                    in0                      0 1 in0

**Explanation**

  1. ``7767517``, it is still the same
  2. ``2029 2547``, we have added an extra layer, so we need to update ``2028`` to ``2029``.
     We don't need to change ``2547`` since the newly added layer has no inputs or outputs.
  3. ``SherpaMetaData  sherpa_meta_data1  0 0 0=2 1=32 2=4 3=7 -23316=5,2,4,3,2,4 -23317=5,384,384,384,384,384 -23318=5,192,192,192,192,192 -23319=5,1,2,4,8,2 -23320=5,31,31,31,31,31``
     This line is newly added. Its explanation is given below:

      - ``SherpaMetaData`` is the type of this layer. Must be ``SherpaMetaData``.
      - ``sherpa_meta_data1`` is the name of this layer. Must be ``sherpa_meta_data1``.
      - ``0 0`` means this layer has no inputs or output. Must be ``0 0``
      - ``0=2``, 0 is the key and 2 is the value. MUST be ``0=2``
      - ``1=32``, 1 is the key and 32 is the value of the
        parameter ``--decode-chunk-len`` that you provided when running
        ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.
      - ``2=4``, 2 is the key and 4 is the value of the
        parameter ``--num-left-chunks`` that you provided when running
        ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.
      - ``3=7``, 3 is the key and 7 is the value of for the amount of padding
        used in the Conv2DSubsampling layer. It should be 7 for zipformer
        if you don't change zipformer.py.
      - ``15=1``, attribute 15, this is the model version. Starting from
        `sherpa-ncnn`_ v2.0, we require that the model version has to
        be >= 1.
      - ``-23316=5,2,4,3,2,4``, attribute 16, this is an array attribute.
        It is attribute 16 since -23300 - (-23316) = 16.
        The first element of the array is the length of the array, which is 5 in our case.
        ``2,4,3,2,4`` is the value of ``--num-encoder-layers``that you provided
        when running ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.
      - ``-23317=5,384,384,384,384,384``, attribute 17.
        The first element of the array is the length of the array, which is 5 in our case.
        ``384,384,384,384,384`` is the value of ``--encoder-dims``that you provided
        when running ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.
      - ``-23318=5,192,192,192,192,192``, attribute 18.
        The first element of the array is the length of the array, which is 5 in our case.
        ``192,192,192,192,192`` is the value of ``--attention-dims`` that you provided
        when running ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.
      - ``-23319=5,1,2,4,8,2``, attribute 19.
        The first element of the array is the length of the array, which is 5 in our case.
        ``1,2,4,8,2`` is the value of ``--zipformer-downsampling-factors`` that you provided
        when running ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.
      - ``-23320=5,31,31,31,31,31``, attribute 20.
        The first element of the array is the length of the array, which is 5 in our case.
        ``31,31,31,31,31`` is the value of ``--cnn-module-kernels`` that you provided
        when running ``./pruned_transducer_stateless7_streaming/export-for-ncnn.py``.

      For ease of reference, we list the key-value pairs that you need to add
      in the following table. If your model has a different setting, please
      change the values for ``SherpaMetaData`` accordingly. Otherwise, you
      will be ``SAD``.

          +----------+--------------------------------------------+
          | key      | value                                      |
          +==========+============================================+
          | 0        | 2 (fixed)                                  |
          +----------+--------------------------------------------+
          | 1        | ``-decode-chunk-len``                      |
          +----------+--------------------------------------------+
          | 2        | ``--num-left-chunks``                      |
          +----------+--------------------------------------------+
          | 3        | 7 (if you don't change code)               |
          +----------+--------------------------------------------+
          | 15       | 1 (The model version)                      |
          +----------+--------------------------------------------+
          |-23316    | ``--num-encoder-layer``                    |
          +----------+--------------------------------------------+
          |-23317    | ``--encoder-dims``                         |
          +----------+--------------------------------------------+
          |-23318    | ``--attention-dims``                       |
          +----------+--------------------------------------------+
          |-23319    | ``--zipformer-downsampling-factors``       |
          +----------+--------------------------------------------+
          |-23320    | ``--cnn-module-kernels``                   |
          +----------+--------------------------------------------+

  4. ``Input in0 0 1 in0``. No need to change it.

.. caution::

  When you add a new layer ``SherpaMetaData``, please remember to update the
  number of layers. In our case, update  ``2028`` to ``2029``. Otherwise,
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
  - ``Android``: `<https://k2-fsa.github.io/sherpa/ncnn/android/index.html>`_
  - ``iOS``: `<https://k2-fsa.github.io/sherpa/ncnn/ios/index.html>`_
  - Python: `<https://k2-fsa.github.io/sherpa/ncnn/python/index.html>`_

We have a list of pre-trained models that have been exported for `sherpa-ncnn`_:

  - `<https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html>`_

    You can find more usages there.
