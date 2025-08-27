Finetune from a pre-trained Zipformer model with adapters
=========================================================

This tutorial shows you how to fine-tune a pre-trained **Zipformer**
transducer model on a new dataset with adapters. 
Adapters are compact and efficient module that can be integrated into a pre-trained model
to improve the model's performance on a new domain. Adapters are injected
between different modules in the well-trained neural network. During training, only the parameters
in the adapters will be updated. It achieves competitive performance
while requiring much less GPU memory than full fine-tuning. For more details about adapters,
please refer to the original `paper <https://arxiv.org/pdf/1902.00751.pdf#/>`_ for more details.

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


Fine-tune with adapter
----------------------

We insert 4 adapters with residual connection in each ``Zipformer2EncoderLayer``. 
The original model parameters remain untouched during training and only the parameters of
the adapters are updated. The following command starts a fine-tuning experiment with adapters:

.. code-block:: bash
    
    $ do_finetune=1
    $ use_adapters=1
    $ adapter_dim=8

    $ ./zipformer_adapter/train.py \
        --world-size 2 \
        --num-epochs 20 \
        --start-epoch 1 \
        --exp-dir zipformer_adapter/exp_giga_finetune_adapters${use_adapters}_adapter_dim${adapter_dim} \
        --use-fp16 1 \
        --base-lr 0.045 \
        --use-adapters $use_adapters --adapter-dim $adapter_dim \
        --bpe-model data/lang_bpe_500/bpe.model \
        --do-finetune $do_finetune \
        --master-port 13022 \
        --finetune-ckpt icefall-asr-librispeech-zipformer-2023-05-15/exp/pretrained.pt \
        --max-duration 1000

The following arguments are related to fine-tuning:

- ``--do-finetune``
    If True, do fine-tuning by initializing the model from a pre-trained checkpoint.
    **Note that if you want to resume your fine-tuning experiment from certain epochs, you
    need to set this to False.**

- ``use-adapters``
    If adapters are used during fine-tuning.

- ``--adapter-dim``
    The bottleneck dimension of the adapter module. Typically a small number.

You should notice that in the training log, the total number of trainale parameters is shown:

.. code-block::

    2024-02-22 21:22:03,808 INFO [train.py:1277] A total of 761344 trainable parameters (1.148% of the whole model)
    
The trainable parameters only makes up 1.15% of the entire model parameters, so the training will be much faster
and requires less memory than full fine-tuning.


Decoding
--------

After training, let's test the WERs. To test the WERs on the GigaSpeech set,
you can execute the following command:

.. code-block:: bash

    $ epoch=20
    $ avg=10
    $ use_adapters=1
    $ adapter_dim=8
    
    % ./zipformer/decode.py \
        --epoch $epoch \
        --avg $avg \
        --use-averaged-model 1 \
        --exp-dir zipformer_adapter/exp_giga_finetune_adapters${use_adapters}_adapter_dim${adapter_dim} \
        --max-duration 600 \
        --use-adapters $use_adapters \
        --adapter-dim $adapter_dim \
        --decoding-method greedy_search

You should see the following numbers:

.. code-block::

    For dev, WER of different settings are:
    greedy_search	15.44	best for dev

    For test, WER of different settings are:
    greedy_search	15.42	best for test


The WER on test set is improved from 19.27 to 15.42, demonstrating the effectiveness of adapters.

The same model can be used to perform decoding on LibriSpeech test sets. You can deactivate the adapters
to keep the same performance of the original model:

.. code-block:: bash

    $ epoch=20
    $ avg=1
    $ use_adapters=0
    $ adapter_dim=8
    
    % ./zipformer/decode.py \
        --epoch $epoch \
        --avg $avg \
        --use-averaged-model 1 \
        --exp-dir zipformer_adapter/exp_giga_finetune_adapters${use_adapters}_adapter_dim${adapter_dim} \
        --max-duration 600 \
        --use-adapters $use_adapters \
        --adapter-dim $adapter_dim \
        --decoding-method greedy_search


.. code-block::

    For dev, WER of different settings are:
    greedy_search	2.23	best for test-clean

    For test, WER of different settings are:
    greedy_search	4.96	best for test-other

The numbers are the same as reported in `icefall <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#normal-scaled-model-number-of-model-parameters-65549011-ie-6555-m>`_. So adapter-based
fine-tuning is also very flexible as the same model can be used for decoding on the original and target domain.


Export the model
----------------

After training, the model can be exported to ``onnx`` format easily using the following command:

.. code-block:: bash

    $ use_adapters=1
    $ adapter_dim=16

    $ ./zipformer_adapter/export-onnx.py \
        --tokens icefall-asr-librispeech-zipformer-2023-05-15/data/lang_bpe_500/tokens.txt \
        --use-averaged-model 1 \
        --epoch 20 \
        --avg 10 \
        --exp-dir zipformer_adapter/exp_giga_finetune_adapters${use_adapters}_adapter_dim${adapter_dim} \
        --use-adapters $use_adapters \
        --adapter-dim $adapter_dim \
        --num-encoder-layers "2,2,3,4,3,2" \
        --downsampling-factor "1,2,4,8,4,2" \
        --feedforward-dim "512,768,1024,1536,1024,768" \
        --num-heads "4,4,4,8,4,4" \
        --encoder-dim "192,256,384,512,384,256" \
        --query-head-dim 32 \
        --value-head-dim 12 \
        --pos-head-dim 4 \
        --pos-dim 48 \
        --encoder-unmasked-dim "192,192,256,256,256,192" \
        --cnn-module-kernel "31,31,15,15,15,31" \
        --decoder-dim 512 \
        --joiner-dim 512 \
        --causal False \
        --chunk-size "16,32,64,-1" \
        --left-context-frames "64,128,256,-1"