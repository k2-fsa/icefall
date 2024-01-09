.. _train_nnlm:

Train an RNN language model
======================================

If you have enough text data, you can train a neural network language model (NNLM) to improve
the WER of your E2E ASR system. This tutorial shows you how to train an RNNLM from
scratch.

.. HINT::

    For how to use an NNLM during decoding, please refer to the following tutorials:
    :ref:`shallow_fusion`, :ref:`LODR`, :ref:`rescoring`

.. note::

    This tutorial is based on the LibriSpeech recipe. Please check it out for the necessary
    python scripts for this tutorial. We use the LibriSpeech LM-corpus as the LM training set
    for illustration purpose. You can also collect your own data. The data format is quite simple:
    each line should contain a complete sentence, and words should be separated by space.

First, let's download the training data for the RNNLM. This can be done via the
following command:

.. code-block:: bash

    $ wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
    $ gzip -d librispeech-lm-norm.txt.gz

As we are training a BPE-level RNNLM, we need to tokenize the training text, which requires a
BPE tokenizer. This can be achieved by executing the following command:

.. code-block:: bash

    $ # if you don't have the BPE
    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
    $ cd icefall-asr-librispeech-zipformer-2023-05-15/data/lang_bpe_500
    $ git lfs pull --include bpe.model
    $ cd ../../..

    $ ./local/prepare_lm_training_data.py \
        --bpe-model icefall-asr-librispeech-zipformer-2023-05-15/data/lang_bpe_500/bpe.model \
        --lm-data librispeech-lm-norm.txt \
        --lm-archive data/lang_bpe_500/lm_data.pt

Now, you should have a file name ``lm_data.pt`` file store under the directory ``data/lang_bpe_500``.
This is the packed training data for the RNNLM. We then sort the training data according to its
sentence length.

.. code-block:: bash

    $ # This could take a while (~ 20 minutes), feel free to grab a cup of coffee :)
    $ ./local/sort_lm_training_data.py \
        --in-lm-data data/lang_bpe_500/lm_data.pt \
        --out-lm-data data/lang_bpe_500/sorted_lm_data.pt \
        --out-statistics data/lang_bpe_500/lm_data_stats.txt


The aforementioned steps can be repeated to create a a validation set for you RNNLM. Let's say
you have a validation set in ``valid.txt``, you can just set ``--lm-data valid.txt``
and ``--lm-archive data/lang_bpe_500/lm-data-valid.pt`` when calling ``./local/prepare_lm_training_data.py``.

After completing the previous steps, the training and testing sets for training RNNLM are ready.
The next step is to train the RNNLM model. The training command is as follows:

.. code-block:: bash

    $ # assume you are in the icefall root directory
    $ cd rnn_lm
    $ ln -s ../../egs/librispeech/ASR/data .
    $ cd ..
    $ ./rnn_lm/train.py \
        --world-size 4 \
        --exp-dir ./rnn_lm/exp \
        --start-epoch 0 \
        --num-epochs 10 \
        --use-fp16 0 \
        --tie-weights 1 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 300 \
        --lm-data rnn_lm/data/lang_bpe_500/sorted_lm_data.pt \
        --lm-data-valid rnn_lm/data/lang_bpe_500/sorted_lm_data.pt


.. note::

    You can adjust the RNNLM hyper parameters to control the size of the RNNLM,
    such as embedding dimension and hidden state dimension. For more details, please
    run ``./rnn_lm/train.py --help``.

.. note::

    The training of RNNLM can take a long time (usually a couple of days).
