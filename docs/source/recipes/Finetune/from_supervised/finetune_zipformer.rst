Finetune from a supervised pre-trained Zipformer model
======================================================

This tutorial shows you how to fine-tune a supervised pre-trained **Zipformer**
transducer model on a new dataset.

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  We recommend you to use a GPU or several GPUs to run this recipe


For illustration purpose, we fine-tune the Zipformer transducer model
pre-trained on `LibriSpeech`_ on the small subset of `GigaSpeech`_. You could use your
own data for fine-tuning if you create a manifest for your new dataset.

Data preparation
----------------

Please follow the instructions in the `GigaSpeech recipe <https://github.com/k2-fsa/icefall/tree/master/egs/gigaspeech/ASR>`_
to prepare the fine-tune data used in this tutorial. We only require the small subset in GigaSpeech for this tutorial.


Model preparation
-----------------

We are using the Zipformer model trained on full LibriSpeech (960 hours) as the intialization. The
checkpoint of the model can be downloaded via the following command:

.. code-block:: bash

    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
    $ cd icefall-asr-librispeech-zipformer-2023-05-15/exp
    $ git lfs pull --include "pretrained.pt"
    $ ln -s pretrained.pt epoch-99.pt
    $ cd ../data/lang_bpe_500
    $ git lfs pull --include bpe.model
    $ cd ../../..

Before fine-tuning, let's test the model's WER on the new domain. The following command performs
decoding on the GigaSpeech test sets:

.. code-block:: bash

    ./zipformer/decode_gigaspeech.py \
        --epoch 99 \
        --avg 1 \
        --exp-dir icefall-asr-librispeech-zipformer-2023-05-15/exp \
        --use-averaged-model 0 \
        --max-duration 1000 \
        --decoding-method greedy_search

You should see the following numbers:

.. code-block::

    For dev, WER of different settings are:
    greedy_search	20.06	best for dev

    For test, WER of different settings are:
    greedy_search	19.27	best for test


Fine-tune
---------

Since LibriSpeech and GigaSpeech are both English dataset, we can initialize the whole
Zipformer model with the checkpoint downloaded in the previous step (otherwise we should consider
initializing the stateless decoder and joiner from scratch due to the mismatch of the output
vocabulary). The following command starts a fine-tuning experiment:

.. code-block:: bash

    $ use_mux=0
    $ do_finetune=1

    $ ./zipformer/finetune.py \
        --world-size 2 \
        --num-epochs 20 \
        --start-epoch 1 \
        --exp-dir zipformer/exp_giga_finetune${do_finetune}_mux${use_mux} \
        --use-fp16 1 \
        --base-lr 0.0045 \
        --bpe-model data/lang_bpe_500/bpe.model \
        --do-finetune $do_finetune \
        --use-mux $use_mux \
        --master-port 13024 \
        --finetune-ckpt icefall-asr-librispeech-zipformer-2023-05-15/exp/pretrained.pt \
        --max-duration 1000

The following arguments are related to fine-tuning:

- ``--base-lr``
    The learning rate used for fine-tuning. We suggest to set a **small** learning rate for fine-tuning,
    otherwise the model may forget the initialization very quickly. A reasonable value should be around
    1/10 of the original lr, i.e 0.0045.

- ``--do-finetune``
    If True, do fine-tuning by initializing the model from a pre-trained checkpoint.
    **Note that if you want to resume your fine-tuning experiment from certain epochs, you
    need to set this to False.**

- ``--finetune-ckpt``
    The path to the pre-trained checkpoint (used for initialization).

- ``--use-mux``
    If True, mix the fine-tune data with the original training data by using `CutSet.mux <https://lhotse.readthedocs.io/en/latest/api.html#lhotse.supervision.SupervisionSet.mux>`_
    This helps maintain the model's performance on the original domain if the original training
    is available. **If you don't have the original training data, please set it to False.**

After fine-tuning, let's test the WERs. You can do this via the following command:

.. code-block:: bash

    $ use_mux=0
    $ do_finetune=1
    $ ./zipformer/decode_gigaspeech.py \
        --epoch 20 \
        --avg 10 \
        --exp-dir zipformer/exp_giga_finetune${do_finetune}_mux${use_mux} \
        --use-averaged-model 1 \
        --max-duration 1000 \
        --decoding-method greedy_search

You should see numbers similar to the ones below:

.. code-block:: text

    For dev, WER of different settings are:
    greedy_search	13.47	best for dev

    For test, WER of different settings are:
    greedy_search	13.66	best for test

Compared to the original checkpoint, the fine-tuned model achieves much lower WERs
on the GigaSpeech test sets.
