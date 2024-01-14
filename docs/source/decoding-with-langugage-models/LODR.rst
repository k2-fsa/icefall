.. _LODR:

LODR for RNN Transducer
=======================


As a type of E2E model, neural transducers are usually considered as having an internal
language model, which learns the language level information on the training corpus.
In real-life scenario, there is often a mismatch between the training corpus and the target corpus space.
This mismatch can be a problem when decoding for neural transducer models with language models as its internal
language can act "against" the external LM. In this tutorial, we show how to use
`Low-order Density Ratio <https://arxiv.org/abs/2203.16776>`_ to alleviate this effect to further improve the performance
of langugae model integration.

.. note::

    This tutorial is based on the recipe
    `pruned_transducer_stateless7_streaming <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_,
    which is a streaming transducer model trained on `LibriSpeech`_.
    However, you can easily apply LODR to other recipes.
    If you encounter any problems, please open an issue here `icefall <https://github.com/k2-fsa/icefall/issues>`__.


.. note::

    For simplicity, the training and testing corpus in this tutorial are the same (`LibriSpeech`_). However,
    you can change the testing set to any other domains (e.g `GigaSpeech`_) and prepare the language models
    using that corpus.

First, let's have a look at some background information. As the predecessor of LODR, Density Ratio (DR) is first proposed `here <https://arxiv.org/abs/2002.11268>`_
to address the language information mismatch between the training
corpus (source domain) and the testing corpus (target domain). Assuming that the source domain and the test domain
are acoustically similar, DR derives the following formular for decoding with Bayes' theorem:

.. math::

    \text{score}\left(y_u|\mathit{x},y\right) =
    \log p\left(y_u|\mathit{x},y_{1:u-1}\right) +
    \lambda_1 \log p_{\text{Target LM}}\left(y_u|\mathit{x},y_{1:u-1}\right) -
    \lambda_2 \log p_{\text{Source LM}}\left(y_u|\mathit{x},y_{1:u-1}\right)


where :math:`\lambda_1` and :math:`\lambda_2` are the weights of LM scores for target domain and source domain respectively.
Here, the source domain LM is trained on the training corpus. The only difference in the above formular compared to
shallow fusion is the subtraction of the source domain LM.

Some works treat the predictor and the joiner of the neural transducer as its internal LM. However, the LM is
considered to be weak and can only capture low-level language information. Therefore, `LODR <https://arxiv.org/abs/2203.16776>`__ proposed to use
a low-order n-gram LM as an approximation of the ILM of the neural transducer. This leads to the following formula
during decoding for transducer model:

.. math::

    \text{score}\left(y_u|\mathit{x},y\right) =
    \log p_{rnnt}\left(y_u|\mathit{x},y_{1:u-1}\right) +
    \lambda_1 \log p_{\text{Target LM}}\left(y_u|\mathit{x},y_{1:u-1}\right) -
    \lambda_2 \log p_{\text{bi-gram}}\left(y_u|\mathit{x},y_{1:u-1}\right)

In LODR, an additional bi-gram LM estimated on the source domain (e.g training corpus) is required. Compared to DR,
the only difference lies in the choice of source domain LM. According to the original `paper <https://arxiv.org/abs/2203.16776>`_,
LODR achieves similar performance compared DR in both intra-domain and cross-domain settings.
As a bi-gram is much faster to evaluate, LODR is usually much faster.

Now, we will show you how to use LODR in ``icefall``.
For illustration purpose, we will use a pre-trained ASR model from this `link <https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29>`_.
If you want to train your model from scratch, please have a look at :ref:`non_streaming_librispeech_pruned_transducer_stateless`.
The testing scenario here is intra-domain (we decode the model trained on `LibriSpeech`_ on `LibriSpeech`_ testing sets).

As the initial step, let's download the pre-trained model.

.. code-block:: bash

    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
    $ cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ git lfs pull --include "pretrained.pt"
    $ ln -s pretrained.pt epoch-99.pt # create a symbolic link so that the checkpoint can be loaded
    $ cd ../data/lang_bpe_500
    $ git lfs pull --include bpe.model
    $ cd ../../..

To test the model, let's have a look at the decoding results **without** using LM. This can be done via the following command:

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

Then, we download the external language model and bi-gram LM that are necessary for LODR.
Note that the bi-gram is estimated on the LibriSpeech 960 hours' text.

.. code-block:: bash

    $ # download the external LM
    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
    $ # create a symbolic link so that the checkpoint can be loaded
    $ pushd icefall-librispeech-rnn-lm/exp
    $ git lfs pull --include "pretrained.pt"
    $ ln -s pretrained.pt epoch-99.pt
    $ popd
    $
    $ # download the bi-gram
    $ git lfs install
    $ git clone https://huggingface.co/marcoyang/librispeech_bigram
    $ pushd data/lang_bpe_500
    $ ln -s ../../librispeech_bigram/2gram.fst.txt .
    $ popd

Then, we perform LODR decoding by setting ``--decoding-method`` to ``modified_beam_search_lm_LODR``:

.. code-block:: bash

    $ exp_dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ lm_dir=./icefall-librispeech-rnn-lm/exp
    $ lm_scale=0.42
    $ LODR_scale=-0.24
    $ ./pruned_transducer_stateless7_streaming/decode.py \
        --epoch 99 \
        --avg 1 \
        --use-averaged-model False \
        --beam-size 4 \
        --exp-dir $exp_dir \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method modified_beam_search_LODR \
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
        --lm-vocab-size 500 \
        --tokens-ngram 2 \
        --ngram-lm-scale $LODR_scale

There are two extra arguments that need to be given when doing LODR. ``--tokens-ngram`` specifies the order of n-gram. As we
are using a bi-gram, we set it to 2. ``--ngram-lm-scale`` is the scale of the bi-gram, it should be a negative number
as we are subtracting the bi-gram's score during decoding.

The decoding results obtained with the above command are shown below:

.. code-block:: text

    $ For test-clean, WER of different settings are:
    $ beam_size_4	2.61	best for test-clean
    $ For test-other, WER of different settings are:
    $ beam_size_4	6.74	best for test-other

Recall that the lowest WER we obtained in :ref:`shallow_fusion` with beam size of 4 is ``2.77/7.08``, LODR
indeed **further improves** the WER. We can do even better if we increase ``--beam-size``:

.. list-table:: WER of LODR with different beam sizes
   :widths: 25 25 50
   :header-rows: 1

   * - Beam size
     - test-clean
     - test-other
   * - 4
     - 2.61
     - 6.74
   * - 8
     - 2.45
     - 6.38
   * - 12
     - 2.4
     - 6.23
