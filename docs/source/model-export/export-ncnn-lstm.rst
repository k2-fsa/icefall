.. _export_lstm_transducer_models_to_ncnn:

Export LSTM transducer models to ncnn
-------------------------------------

We use the pre-trained model from the following repository as an example:

`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03>`_

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
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

  git lfs pull --include "exp/pretrained-iter-468000-avg-16.pt"
  git lfs pull --include "data/lang_bpe_500/bpe.model"

  cd ..

.. note::

  We downloaded ``exp/pretrained-xxx.pt``, not ``exp/cpu-jit_xxx.pt``.

In the above code, we downloaded the pre-trained model into the directory
``egs/librispeech/ASR/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03``.

2. Install ncnn and pnnx
^^^^^^^^^^^^^^^^^^^^^^^^

Please refer to :ref:`export_for_ncnn_install_ncnn_and_pnnx` .


3. Export the model via torch.jit.trace()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let us rename our pre-trained model:

.. code-block::

  cd egs/librispeech/ASR

  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp

  ln -s pretrained-iter-468000-avg-16.pt epoch-99.pt

  cd ../..

Next, we use the following code to export our model:

.. code-block:: bash

  dir=./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

  ./lstm_transducer_stateless2/export-for-ncnn.py \
    --exp-dir $dir/exp \
    --tokens $dir/data/lang_bpe_500/tokens.txt \
    --epoch 99 \
    --avg 1 \
    --use-averaged-model 0 \
    --num-encoder-layers 12 \
    --encoder-dim 512 \
    --rnn-hidden-size 1024

.. hint::

  We have renamed our model to ``epoch-99.pt`` so that we can use ``--epoch 99``.
  There is only one pre-trained model, so we use ``--avg 1 --use-averaged-model 0``.

  If you have trained a model by yourself and if you have all checkpoints
  available, please first use ``decode.py`` to tune ``--epoch --avg``
  and select the best combination with with ``--use-averaged-model 1``.

.. note::

  You will see the following log output:

  .. literalinclude:: ./code/export-lstm-transducer-for-ncnn-output.txt

  The log shows the model has ``84176356`` parameters, i.e., ``~84 M``.

  .. code-block::

    ls -lh icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/pretrained-iter-468000-avg-16.pt

    -rw-r--r-- 1 kuangfangjun root 324M Feb 17 10:34 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/pretrained-iter-468000-avg-16.pt

  You can see that the file size of the pre-trained model is ``324 MB``, which
  is roughly equal to ``84176356*4/1024/1024 = 321.107 MB``.

After running ``lstm_transducer_stateless2/export-for-ncnn.py``,
we will get the following files:

.. code-block:: bash

  ls -lh icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/*pnnx.pt

  -rw-r--r-- 1 kuangfangjun root 1010K Feb 17 11:22 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.pt
  -rw-r--r-- 1 kuangfangjun root  318M Feb 17 11:22 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.pt
  -rw-r--r-- 1 kuangfangjun root  3.0M Feb 17 11:22 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.pt


.. _lstm-transducer-step-4-export-torchscript-model-via-pnnx:

4. Export torchscript model via pnnx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

  Make sure you have set up the ``PATH`` environment variable
  in :ref:`export_for_ncnn_install_ncnn_and_pnnx`. Otherwise,
  it will throw an error saying that ``pnnx`` could not be found.

Now, it's time to export our models to `ncnn`_ via ``pnnx``.

.. code-block::

  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/

  pnnx ./encoder_jit_trace-pnnx.pt
  pnnx ./decoder_jit_trace-pnnx.pt
  pnnx ./joiner_jit_trace-pnnx.pt

It will generate the following files:

.. code-block:: bash

  ls -lh  icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/*ncnn*{bin,param}

  -rw-r--r-- 1 kuangfangjun root 503K Feb 17 11:32 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  437 Feb 17 11:32 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 159M Feb 17 11:32 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  21K Feb 17 11:32 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 1.5M Feb 17 11:33 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  488 Feb 17 11:33 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.ncnn.param


There are two types of files:

- ``param``: It is a text file containing the model architectures. You can
  use a text editor to view its content.
- ``bin``: It is a binary file containing the model parameters.

We compare the file sizes of the models below before and after converting via ``pnnx``:

.. see https://tableconvert.com/restructuredtext-generator

+----------------------------------+------------+
| File name                        | File size  |
+==================================+============+
| encoder_jit_trace-pnnx.pt        | 318 MB     |
+----------------------------------+------------+
| decoder_jit_trace-pnnx.pt        | 1010 KB    |
+----------------------------------+------------+
| joiner_jit_trace-pnnx.pt         | 3.0 MB     |
+----------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin  | 159 MB     |
+----------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin  | 503 KB     |
+----------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin   | 1.5 MB     |
+----------------------------------+------------+

You can see that the file sizes of the models after conversion are about one half
of the models before conversion:

  - encoder: 318 MB vs 159 MB
  - decoder: 1010 KB vs 503 KB
  - joiner: 3.0 MB vs 1.5 MB

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

  python3 ./lstm_transducer_stateless2/streaming-ncnn-decode.py \
    --tokens ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/data/lang_bpe_500/tokens.txt \
    --encoder-param-filename ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.ncnn.param \
    --encoder-bin-filename ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.ncnn.bin \
    --decoder-param-filename ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.ncnn.param \
    --decoder-bin-filename ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.ncnn.bin \
    --joiner-param-filename ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.ncnn.param \
    --joiner-bin-filename ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.ncnn.bin \
    ./icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/test_wavs/1089-134686-0001.wav

.. hint::

  `ncnn`_ supports only ``batch size == 1``, so ``streaming-ncnn-decode.py`` accepts
  only 1 wave file as input.

The output is given below:

.. literalinclude:: ./code/test-streaming-ncnn-decode-lstm-transducer-libri.txt

Congratulations! You have successfully exported a model from PyTorch to `ncnn`_!

.. _lstm-modify-the-exported-encoder-for-sherpa-ncnn:

6. Modify the exported encoder for sherpa-ncnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use the exported models in `sherpa-ncnn`_, we have to modify
``encoder_jit_trace-pnnx.ncnn.param``.

Let us have a look at the first few lines of ``encoder_jit_trace-pnnx.ncnn.param``:

.. code-block::

  7767517
  267 379
  Input                    in0                      0 1 in0

**Explanation** of the above three lines:

  1. ``7767517``, it is a magic number and should not be changed.
  2. ``267 379``, the first number ``267`` specifies the number of layers
     in this file, while ``379`` specifies the number of intermediate outputs
     of this file
  3. ``Input in0 0 1 in0``, ``Input`` is the layer type of this layer; ``in0``
     is the layer name of this layer; ``0`` means this layer has no input;
     ``1`` means this layer has one output; ``in0`` is the output name of
     this layer.

We need to add 1 extra line and also increment the number of layers.
The result looks like below:

.. code-block:: bash

  7767517
  268 379
  SherpaMetaData           sherpa_meta_data1        0 0 0=3 1=12 2=512 3=1024
  Input                    in0                      0 1 in0

**Explanation**

  1. ``7767517``, it is still the same
  2. ``268 379``, we have added an extra layer, so we need to update ``267`` to ``268``.
     We don't need to change ``379`` since the newly added layer has no inputs or outputs.
  3. ``SherpaMetaData  sherpa_meta_data1  0 0 0=3 1=12 2=512 3=1024``
     This line is newly added. Its explanation is given below:

      - ``SherpaMetaData`` is the type of this layer. Must be ``SherpaMetaData``.
      - ``sherpa_meta_data1`` is the name of this layer. Must be ``sherpa_meta_data1``.
      - ``0 0`` means this layer has no inputs or output. Must be ``0 0``
      - ``0=3``, 0 is the key and 3 is the value. MUST be ``0=3``
      - ``1=12``, 1 is the key and 12 is the value of the
        parameter ``--num-encoder-layers`` that you provided when running
        ``./lstm_transducer_stateless2/export-for-ncnn.py``.
      - ``2=512``, 2 is the key and 512 is the value of the
        parameter ``--encoder-dim`` that you provided when running
        ``./lstm_transducer_stateless2/export-for-ncnn.py``.
      - ``3=1024``, 3 is the key and 1024 is the value of the
        parameter ``--rnn-hidden-size`` that you provided when running
        ``./lstm_transducer_stateless2/export-for-ncnn.py``.

      For ease of reference, we list the key-value pairs that you need to add
      in the following table. If your model has a different setting, please
      change the values for ``SherpaMetaData`` accordingly. Otherwise, you
      will be ``SAD``.

          +------+-----------------------------+
          | key  | value                       |
          +======+=============================+
          | 0    | 3 (fixed)                   |
          +------+-----------------------------+
          | 1    | ``--num-encoder-layers``    |
          +------+-----------------------------+
          | 2    | ``--encoder-dim``           |
          +------+-----------------------------+
          | 3    | ``--rnn-hidden-size``       |
          +------+-----------------------------+

  4. ``Input in0 0 1 in0``. No need to change it.

.. caution::

  When you add a new layer ``SherpaMetaData``, please remember to update the
  number of layers. In our case, update  ``267`` to ``268``. Otherwise,
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

7. (Optional) int8 quantization with sherpa-ncnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is optional.

In this step, we describe how to quantize our model with ``int8``.

Change :ref:`lstm-transducer-step-4-export-torchscript-model-via-pnnx` to
disable ``fp16`` when using ``pnnx``:

.. code-block::

  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/

  pnnx ./encoder_jit_trace-pnnx.pt fp16=0
  pnnx ./decoder_jit_trace-pnnx.pt
  pnnx ./joiner_jit_trace-pnnx.pt fp16=0

.. note::

  We add ``fp16=0`` when exporting the encoder and joiner. `ncnn`_ does not
  support quantizing the decoder model yet. We will update this documentation
  once `ncnn`_ supports it. (Maybe in this year, 2023).

.. code-block:: bash

  ls -lh icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/*_jit_trace-pnnx.ncnn.{param,bin}

  -rw-r--r-- 1 kuangfangjun root 503K Feb 17 11:32 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  437 Feb 17 11:32 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 317M Feb 17 11:54 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  21K Feb 17 11:54 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-pnnx.ncnn.param
  -rw-r--r-- 1 kuangfangjun root 3.0M Feb 17 11:54 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.ncnn.bin
  -rw-r--r-- 1 kuangfangjun root  488 Feb 17 11:54 icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-pnnx.ncnn.param


Let us compare again the file sizes:

+----------------------------------------+------------+
| File name                              | File size  |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.pt              | 318 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.pt              | 1010 KB    |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.pt               | 3.0 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp16) | 159 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin (fp16) | 503 KB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp16) | 1.5 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp32) | 317 MB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp32) | 3.0 MB     |
+----------------------------------------+------------+

You can see that the file sizes are doubled when we disable ``fp16``.

.. note::

  You can again use ``streaming-ncnn-decode.py`` to test the exported models.

Next, follow :ref:`lstm-modify-the-exported-encoder-for-sherpa-ncnn`
to modify ``encoder_jit_trace-pnnx.ncnn.param``.

Change

.. code-block:: bash

  7767517
  267 379
  Input                    in0                      0 1 in0

to

.. code-block:: bash

  7767517
  268 379
  SherpaMetaData           sherpa_meta_data1        0 0 0=3 1=12 2=512 3=1024
  Input                    in0                      0 1 in0

.. caution::

  Please follow :ref:`lstm-modify-the-exported-encoder-for-sherpa-ncnn`
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
from the pre-trained model repository
`<https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03>`_

.. code-block:: bash

  cd egs/librispeech/ASR
  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/

  cat <<EOF > wave_filenames.txt
  ../test_wavs/1089-134686-0001.wav
  ../test_wavs/1221-135766-0001.wav
  ../test_wavs/1221-135766-0002.wav
  EOF

Now we can calculate the scales needed for quantization with the calibration data:

.. code-block:: bash

  cd egs/librispeech/ASR
  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/

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

.. literalinclude:: ./code/generate-int-8-scale-table-for-lstm.txt

It generates the following two files:

.. code-block:: bash

  ls -lh encoder-scale-table.txt joiner-scale-table.txt

  -rw-r--r-- 1 kuangfangjun root 345K Feb 17 12:13 encoder-scale-table.txt
  -rw-r--r-- 1 kuangfangjun root  17K Feb 17 12:13 joiner-scale-table.txt

.. caution::

  Definitely, you need more calibration data to compute the scale table.

Finally, let us use the scale table to quantize our models into ``int8``.

.. code-block:: bash

  ncnn2int8

  usage: ncnn2int8 [inparam] [inbin] [outparam] [outbin] [calibration table]

First, we quantize the encoder model:

.. code-block:: bash

  cd egs/librispeech/ASR
  cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/

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

.. code-block::

  -rw-r--r-- 1 kuangfangjun root 218M Feb 17 12:19 encoder_jit_trace-pnnx.ncnn.int8.bin
  -rw-r--r-- 1 kuangfangjun root  21K Feb 17 12:19 encoder_jit_trace-pnnx.ncnn.int8.param
  -rw-r--r-- 1 kuangfangjun root 774K Feb 17 12:19 joiner_jit_trace-pnnx.ncnn.int8.bin
  -rw-r--r-- 1 kuangfangjun root  496 Feb 17 12:19 joiner_jit_trace-pnnx.ncnn.int8.param

Congratulations! You have successfully quantized your model from ``float32`` to ``int8``.

.. caution::

  ``ncnn.int8.param`` and ``ncnn.int8.bin`` must be used in pairs.

  You can replace ``ncnn.param`` and ``ncnn.bin`` with ``ncnn.int8.param``
  and ``ncnn.int8.bin`` in `sherpa-ncnn`_ if you like.

  For instance, to use only the ``int8`` encoder in ``sherpa-ncnn``, you can
  replace the following invocation:

    .. code-block::

      cd egs/librispeech/ASR
      cd icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/

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

    .. code-block:: bash

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
| encoder_jit_trace-pnnx.pt              | 318 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.pt              | 1010 KB    |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.pt               | 3.0 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp16) | 159 MB     |
+----------------------------------------+------------+
| decoder_jit_trace-pnnx.ncnn.bin (fp16) | 503 KB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp16) | 1.5 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.bin (fp32) | 317 MB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.bin  (fp32) | 3.0 MB     |
+----------------------------------------+------------+
| encoder_jit_trace-pnnx.ncnn.int8.bin   | 218 MB     |
+----------------------------------------+------------+
| joiner_jit_trace-pnnx.ncnn.int8.bin    | 774 KB     |
+----------------------------------------+------------+

You can see that the file size of the joiner model after ``int8`` quantization
is much smaller. However, the size of the encoder model is even larger than
the ``fp16`` counterpart. The reason is that `ncnn`_ currently does not support
quantizing ``LSTM`` layers into ``8-bit``. Please see
`<https://github.com/Tencent/ncnn/issues/4532>`_

.. hint::

    Currently, only linear layers and convolutional layers are quantized
    with ``int8``, so you don't see an exact ``4x`` reduction in file sizes.

.. note::

  You need to test the recognition accuracy after ``int8`` quantization.


That's it! Have fun with `sherpa-ncnn`_!
