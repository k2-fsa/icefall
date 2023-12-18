.. _rescoring:

LM rescoring for Transducer
=================================

LM rescoring is a commonly used approach to incorporate external LM information. Unlike shallow-fusion-based
methods (see :ref:`shallow_fusion`, :ref:`LODR`), rescoring is usually performed to re-rank the n-best hypotheses after beam search.
Rescoring is usually more efficient than shallow fusion since less computation is performed on the external LM.
In this tutorial, we will show you how to use external LM to rescore the n-best hypotheses decoded from neural transducer models in
`icefall <https://github.com/k2-fsa/icefall>`__.

.. note::

    This tutorial is based on the recipe 
    `pruned_transducer_stateless7_streaming <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming>`_,
    which is a streaming transducer model trained on `LibriSpeech`_. 
    However, you can easily apply shallow fusion to other recipes.
    If you encounter any problems, please open an issue `here <https://github.com/k2-fsa/icefall/issues>`_.

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

As usual, we first test the model's performance without external LM. This can be done via the following command:

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

Now, we will try to improve the above WER numbers via external LM rescoring. We will download 
a pre-trained LM from this `link <https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm>`__.

.. note::

    This is an RNN LM trained on the LibriSpeech text corpus. So it might not be ideal for other corpus.
    You may also train a RNN LM from scratch. Please refer to this `script <https://github.com/k2-fsa/icefall/blob/master/icefall/rnn_lm/train.py>`__
    for training a RNN LM and this `script <https://github.com/k2-fsa/icefall/blob/master/icefall/transformer_lm/train.py>`__ to train a transformer LM.

.. code-block:: bash

    $ # download the external LM
    $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm 
    $ # create a symbolic link so that the checkpoint can be loaded
    $ pushd icefall-librispeech-rnn-lm/exp
    $ git lfs pull --include "pretrained.pt"
    $ ln -s pretrained.pt epoch-99.pt 
    $ popd


With the RNNLM available, we can rescore the n-best hypotheses generated from `modified_beam_search`. Here,
`n` should be the number of beams, i.e ``--beam-size``. The command for LM rescoring is
as follows. Note that the ``--decoding-method`` is set to `modified_beam_search_lm_rescore` and ``--use-shallow-fusion``
is set to `False`.

.. code-block:: bash
    
    $ exp_dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ lm_dir=./icefall-librispeech-rnn-lm/exp
    $ lm_scale=0.43
    $ ./pruned_transducer_stateless7_streaming/decode.py \
        --epoch 99 \
        --avg 1 \
        --use-averaged-model False \
        --beam-size 4 \
        --exp-dir $exp_dir \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method modified_beam_search_lm_rescore \
        --bpe-model ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/bpe.model \
        --use-shallow-fusion 0 \
        --lm-type rnn \
        --lm-exp-dir $lm_dir \
        --lm-epoch 99 \
        --lm-scale $lm_scale \
        --lm-avg 1 \
        --rnn-lm-embedding-dim 2048 \
        --rnn-lm-hidden-dim 2048 \
        --rnn-lm-num-layers 3 \
        --lm-vocab-size 500

.. code-block:: text

    $ For test-clean, WER of different settings are:
    $ beam_size_4	2.93	best for test-clean
    $ For test-other, WER of different settings are:
    $ beam_size_4	7.6	best for test-other

Great! We made some improvements! Increasing the size of the n-best hypotheses will further boost the performance,
see the following table:

.. list-table:: WERs of LM rescoring with different beam sizes
   :widths: 25 25 25
   :header-rows: 1

   * - Beam size
     - test-clean
     - test-other
   * - 4
     - 2.93
     - 7.6
   * - 8
     - 2.67
     - 7.11
   * - 12
     - 2.59
     - 6.86

In fact, we can also apply LODR (see :ref:`LODR`) when doing LM rescoring. To do so, we need to 
download the bi-gram required by LODR:

.. code-block:: bash

    $ # download the bi-gram
    $ git lfs install
    $ git clone https://huggingface.co/marcoyang/librispeech_bigram
    $ pushd data/lang_bpe_500
    $ ln -s ../../librispeech_bigram/2gram.arpa .
    $ popd

Then we can performn LM rescoring + LODR by changing the decoding method to `modified_beam_search_lm_rescore_LODR`. 

.. note:: 

    This decoding method requires the dependency of `kenlm <https://github.com/kpu/kenlm>`_. You can install it
    via this command: `pip install https://github.com/kpu/kenlm/archive/master.zip`. 

.. code-block:: bash
    
    $ exp_dir=./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
    $ lm_dir=./icefall-librispeech-rnn-lm/exp
    $ lm_scale=0.43
    $ ./pruned_transducer_stateless7_streaming/decode.py \
        --epoch 99 \
        --avg 1 \
        --use-averaged-model False \
        --beam-size 4 \
        --exp-dir $exp_dir \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method modified_beam_search_lm_rescore_LODR \
        --bpe-model ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/bpe.model \
        --use-shallow-fusion 0 \
        --lm-type rnn \
        --lm-exp-dir $lm_dir \
        --lm-epoch 99 \
        --lm-scale $lm_scale \
        --lm-avg 1 \
        --rnn-lm-embedding-dim 2048 \
        --rnn-lm-hidden-dim 2048 \
        --rnn-lm-num-layers 3 \
        --lm-vocab-size 500

You should see the following WERs after executing the commands above:

.. code-block:: text

    $ For test-clean, WER of different settings are:
    $ beam_size_4	2.9	best for test-clean
    $ For test-other, WER of different settings are:
    $ beam_size_4	7.57	best for test-other

It's slightly better than LM rescoring. If we further increase the beam size, we will see
further improvements from LM rescoring + LODR:

.. list-table:: WERs of LM rescoring + LODR with different beam sizes
   :widths: 25 25 25
   :header-rows: 1

   * - Beam size
     - test-clean
     - test-other
   * - 4
     - 2.9
     - 7.57
   * - 8
     - 2.63
     - 7.04
   * - 12
     - 2.52
     - 6.73

As mentioned earlier, LM rescoring is usually faster than shallow-fusion based methods.
Here, we benchmark the WERs and decoding speed of them:

.. list-table:: LM-rescoring-based methods vs shallow-fusion-based methods (The numbers in each field is WER on test-clean, WER on test-other and decoding time on test-clean)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Decoding method
     - beam=4
     - beam=8
     - beam=12
   * - ``modified_beam_search``
     - 3.11/7.93; 132s
     - 3.1/7.95; 177s
     - 3.1/7.96; 210s
   * - ``modified_beam_search_lm_shallow_fusion``
     - 2.77/7.08; 262s
     - 2.62/6.65; 352s
     - 2.58/6.65; 488s
   * - ``modified_beam_search_LODR``
     - 2.61/6.74; 400s
     - 2.45/6.38; 610s
     - 2.4/6.23; 870s
   * - ``modified_beam_search_lm_rescore``
     - 2.93/7.6; 156s
     - 2.67/7.11; 203s
     - 2.59/6.86; 255s
   * - ``modified_beam_search_lm_rescore_LODR``
     - 2.9/7.57; 160s
     - 2.63/7.04; 203s
     - 2.52/6.73; 263s

.. note::

    Decoding is performed with a single 32G V100, we set ``--max-duration`` to 600. 
    Decoding time here is only for reference and it may vary.