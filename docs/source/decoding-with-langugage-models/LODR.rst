.. _LODR:

LODR for RNN Transducer
=======================


As a type of E2E model, neural transducers are usually considered as having an internal 
language model, which learns the language level information on the training corpus. 
In real-life scenario, there is often a mismatch between the training corpus and the target corpus space. 
This mismatch can be a problem when decoding for neural transducer models with language models as its internal
language can act "against" the external LM. In this tutorial, we show how to use
`Low-order Density Ratio <>`_ to alleviate this effect to further improve the performance
of langugae model integration. 

.. note::

    This tutorial is based on the recipe 
    `pruned_transducer_stateless7_streaming <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_,
    which is a streaming transducer model trained on `LibriSpeech`_. 
    However, you can easily apply shallow fusion to other recipes.
    If you encounter any problems, please open an issue here `icefall <https://github.com/k2-fsa/icefall/issues>`_.


.. note::

    For simplicity, the training and testing corpus in this tutorial is the same (`LibriSpeech`_). However, 
    you can change the testing set to any other domains (e.g GigaSpeech) and prepare the language models 
    using that corpus.

First, let's have a look at some background information. As the predecessor of LODR, Density Ratio (DR) is first proposed `here <https://arxiv.org/abs/2002.11268>`_ 
to address the language information mismatch between the training
corpus (source domain) and the testing corpus (target domain). Assuming that the source domain and the test domain
are acoustically similar, DR derives the following formular for decoding with Bayes' theorem:

.. math::

    \text{score}\left(y_u|\mathit{x},y\right) = 
    \log p\left(y_u|\mathit{x},y_{1:u-1}\right) + 
    \lambda_1 \log p_{\text{source LM}}\left(y_u|\mathit{x},y_{1:u-1}\right) - 
    \lambda_2 \log p_{\text{target LM}}\left(y_u|\mathit{x},y_{1:u-1}\right)


where :math:`\lambda_1` and :math:`\lambda_2` are the LM score for source domain and target domain respectively. 
Here, the source domain LM is trained on the training corpus. The only difference in the above formular compared to 
shallow fusion is the subtraction of the source domain LM.

Some works treat the predictor and the joiner of the neural transducer as its internal LM. However, the LM is 
considered to be weak and can only capture low-level language information. Therefore, `LODR <https://arxiv.org/abs/2203.16776>`_ propose to use
a low-order n-gram LM as an approximation of the ILM of the neural transducer. This leads to the following formula
during decoding for RNNT model:

.. math::

    \text{score}\left(y_u|\mathit{x},y\right) = 
    \log p_{rnnt}\left(y_u|\mathit{x},y_{1:u-1}\right) + 
    \lambda_1 \log p_{\text{LM}}\left(y_u|\mathit{x},y_{1:u-1}\right) - 
    \lambda_2 \log p_{\text{bi-gram}}\left(y_u|\mathit{x},y_{1:u-1}\right)

In LODR, an additional bi-gram LM estimated on the training corpus is required apart from the neural LM. Comared to DR, 
the only difference lies in the choice of source domain LM. According to the original `paper <https://arxiv.org/abs/2203.16776>`_,
LODR achieves similar performance compared DR. As a bi-gram is much faster to evaluate, LODR
is a suitable decoding method for faster inference.


Now, we will show you how to use LODR in ``icefall``.
For illustration purpose, we will use a pre-trained ASR model from this `link <https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29>`_.
If you want to train your model from scratch, please have a look at :ref:`non_streaming_librispeech_pruned_transducer_stateless`.

As the initial step, let's download the pre-trained model.

.. code-block:: bash

    $ git lfs install
    $ git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
    $ pushd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ ln -s pretrained.pt epoch-99.pt # create a symbolic link so that the checkpoint can be loaded

To test the model, let's have a look at the decoding results without using LM. This can be done via the following command:

.. code-block:: bash

    $ exp_dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/
    $ ./pruned_transducer_stateless7_streaming/decode.py \
        --epoch 30 \
        --avg 9 \
        --exp-dir $exp_dir \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method modified_beam_search

The following WERs are achieved on test-clean and test-other:

.. code-block:: bash

    $ For test-clean, WER of different settings are:
    $ beam_size_4	3.11	best for test-clean
    $ For test-other, WER of different settings are:
    $ beam_size_4	7.93	best for test-other


