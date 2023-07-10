Introduction
============

This page shows you how we implement streaming **X-former transducer** models for ASR.

.. HINT::
   X-former transducer here means the encoder of the transducer model uses Multi-Head Attention,
   like `Conformer <https://arxiv.org/pdf/2005.08100.pdf>`_, `EmFormer <https://arxiv.org/pdf/2010.10759.pdf>`_ etc.

Currently we have implemented two types of streaming models, one uses Conformer as encoder, the other uses Emformer as encoder.

Streaming Conformer
-------------------

The main idea of training a streaming model is to make the model see limited contexts
in training time, we can achieve this by applying a mask to the output of self-attention.
In icefall, we implement the streaming conformer the way just like what `WeNet <https://arxiv.org/pdf/2012.05481.pdf>`_ did.

.. NOTE::
   The conformer-transducer recipes in LibriSpeech datasets, like, `pruned_transducer_stateless <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless>`_,
   `pruned_transducer_stateless2 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless2>`_,
   `pruned_transducer_stateless3 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless3>`_,
   `pruned_transducer_stateless4 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless4>`_,
   `pruned_transducer_stateless5 <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless5>`_
   all support streaming.

.. NOTE::
   Training a streaming conformer model in ``icefall`` is almost the same as training a
   non-streaming model, all you need to do is passing several extra arguments.
   See :doc:`Pruned transducer statelessX <librispeech/pruned_transducer_stateless>` for more details.

.. HINT::
   If you want to modify a non-streaming conformer recipe to support both streaming and non-streaming, please refer
   to `this pull request <https://github.com/k2-fsa/icefall/pull/454>`_.  After adding the code needed by streaming training,
   you have to re-train it with the extra arguments metioned in the docs above to get a streaming model.


Streaming Emformer
------------------

The Emformer model proposed `here <https://arxiv.org/pdf/2010.10759.pdf>`_ uses more
complicated techniques. It has a memory bank component to memorize history information,
what' more, it also introduces right context in training time by hard-copying part of
the input features.

We have three variants of Emformer models in ``icefall``.

 - ``pruned_stateless_emformer_rnnt2`` using Emformer from torchaudio, see `LibriSpeech recipe <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_stateless_emformer_rnnt2>`__.
 - ``conv_emformer_transducer_stateless`` using ConvEmformer implemented by ourself. Different from the Emformer in torchaudio,
   ConvEmformer has a convolution in each layer and uses the mechanisms in our reworked conformer model.
   See `LibriSpeech recipe <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless>`__.
 - ``conv_emformer_transducer_stateless2`` using ConvEmformer implemented by ourself. The only difference from the above one is that
   it uses a simplified memory bank. See `LibriSpeech recipe <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2>`_.
