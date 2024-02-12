.. _shallow_fusion:

Shallow fusion for Transducer
=================================

External language models (LM) are commonly used to improve WERs for E2E ASR models.
This tutorial shows you how to perform ``shallow fusion`` with an external LM
to improve the word-error-rate of a transducer model.

.. note::

    This tutorial is based on the recipe
    `pruned_transducer_stateless7_streaming <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_,
    which is a streaming transducer model trained on `LibriSpeech`_.
    However, you can easily apply shallow fusion to other recipes.
    If you encounter any problems, please open an issue here `icefall <https://github.com/k2-fsa/icefall/issues>`_.

.. note::

    For simplicity, the training and testing corpus in this tutorial is the same (`LibriSpeech`_). However, you can change the testing set
    to any other domains (e.g `GigaSpeech`_) and use an external LM trained on that domain.

.. HINT::

  We recommend you to use a GPU for decoding.

For illustration purpose, we will use a pre-trained ASR model from this `link <https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29>`__.
If you want to train your model from scratch, please have a look at :ref:`non_streaming_librispeech_pruned_transducer_stateless`.

As the initial step, let's download the pre-trained model.

.. code-block:: bash

    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
    $ cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ git lfs pull --include "pretrained.pt"
    $ ln -s pretrained.pt epoch-99.pt # create a symbolic link so that the checkpoint can be loaded
    $ cd ../data/lang_bpe_500
    $ git lfs pull --include bpe.model
    $ cd ../../..

To test the model, let's have a look at the decoding results without using LM. This can be done via the following command:

.. code-block:: bash

    $ exp_dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/
    $ ./pruned_transducer_stateless7_streaming/decode.py \
        --epoch 99 \
        --avg 1 \
        --use-averaged-model False \
        --exp-dir $exp_dir \
        --bpe-model ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/bpe.model \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method modified_beam_search

The following WERs are achieved on test-clean and test-other:

.. code-block:: text

    $ For test-clean, WER of different settings are:
    $ beam_size_4	3.11	best for test-clean
    $ For test-other, WER of different settings are:
    $ beam_size_4	7.93	best for test-other

These are already good numbers! But we can further improve it by using shallow fusion with external LM.
Training a language model usually takes a long time, we can download a pre-trained LM from this `link <https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm>`__.

.. code-block:: bash

    $ # download the external LM
    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
    $ # create a symbolic link so that the checkpoint can be loaded
    $ pushd icefall-librispeech-rnn-lm/exp
    $ git lfs pull --include "pretrained.pt"
    $ ln -s pretrained.pt epoch-99.pt
    $ popd

.. note::

    This is an RNN LM trained on the LibriSpeech text corpus. So it might not be ideal for other corpus.
    You may also train a RNN LM from scratch. Please refer to this `script <https://github.com/k2-fsa/icefall/blob/master/icefall/rnn_lm/train.py>`__
    for training a RNN LM and this `script <https://github.com/k2-fsa/icefall/blob/master/icefall/transformer_lm/train.py>`__ to train a transformer LM.

To use shallow fusion for decoding, we can execute the following command:

.. code-block:: bash

    $ exp_dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ lm_dir=./icefall-librispeech-rnn-lm/exp
    $ lm_scale=0.29
    $ ./pruned_transducer_stateless7_streaming/decode.py \
        --epoch 99 \
        --avg 1 \
        --use-averaged-model False \
        --beam-size 4 \
        --exp-dir $exp_dir \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --bpe-model ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/bpe.model \
        --use-shallow-fusion 1 \
        --lm-type rnn \
        --lm-exp-dir $lm_dir \
        --lm-epoch 99 \
        --lm-scale $lm_scale \
        --lm-avg 1 \
        --rnn-lm-embedding-dim 2048 \
        --rnn-lm-hidden-dim 2048 \
        --rnn-lm-num-layers 3 \
        --lm-vocab-size 500

Note that we set ``--decoding-method modified_beam_search_lm_shallow_fusion`` and ``--use-shallow-fusion True``
to use shallow fusion. ``--lm-type`` specifies the type of neural LM we are going to use, you can either choose
between ``rnn`` or ``transformer``. The following three arguments are associated with the rnn:

- ``--rnn-lm-embedding-dim``
    The embedding dimension of the RNN LM

- ``--rnn-lm-hidden-dim``
    The hidden dimension of the RNN LM

- ``--rnn-lm-num-layers``
    The number of RNN layers in the RNN LM.


The decoding result obtained with the above command are shown below.

.. code-block:: text

    $ For test-clean, WER of different settings are:
    $ beam_size_4	2.77	best for test-clean
    $ For test-other, WER of different settings are:
    $ beam_size_4	7.08	best for test-other

The improvement of shallow fusion is very obvious! The relative WER reduction on test-other is around 10.5%.
A few parameters can be tuned to further boost the performance of shallow fusion:

- ``--lm-scale``

    Controls the scale of the LM. If too small, the external language model may not be fully utilized; if too large,
    the LM score might be dominant during decoding, leading to bad WER. A typical value of this is around 0.3.

- ``--beam-size``

    The number of active paths in the search beam. It controls the trade-off between decoding efficiency and accuracy.

Here, we also show how `--beam-size` effect the WER and decoding time:

.. list-table:: WERs and decoding time (on test-clean) of shallow fusion with different beam sizes
   :widths: 25 25 25 25
   :header-rows: 1

   * - Beam size
     - test-clean
     - test-other
     - Decoding time on test-clean (s)
   * - 4
     - 2.77
     - 7.08
     - 262
   * - 8
     - 2.62
     - 6.65
     - 352
   * - 12
     - 2.58
     - 6.65
     - 488

As we see, a larger beam size during shallow fusion improves the WER, but is also slower.








