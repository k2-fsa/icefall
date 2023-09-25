Decoding with language models
=============================

This section describes how to use external langugage models 
during decoding to improve the WER of transducer models. To train an external language model,
please refer to this tutorial: :ref:`train_nnlm`.

The following decoding methods with external langugage models are available:


.. list-table:: 
   :widths: 25 50
   :header-rows: 1

   * - Decoding method
     - beam=4
   * - ``modified_beam_search``
     - Beam search (i.e. really n-best decoding, the "beam" is the value of n), similar to the original RNN-T paper. Note, this method does not use language model. 
   * - ``modified_beam_search_lm_shallow_fusion``
     - As ``modified_beam_search``, but interpolate RNN-T scores with language model scores, also known as shallow fusion
   * - ``modified_beam_search_LODR``
     - As ``modified_beam_search_lm_shallow_fusion``, but subtract score of a (BPE-symbol-level) bigram backoff language model used as an approximation to the internal language model of RNN-T.
   * - ``modified_beam_search_lm_rescore``
     - As ``modified_beam_search``, but rescore the n-best hypotheses with external language model (e.g. RNNLM) and re-rank them.
   * - ``modified_beam_search_lm_rescore_LODR``
     - As ``modified_beam_search_lm_rescore``, but also subtract the score of a (BPE-symbol-level) bigram backoff language model during re-ranking.


.. toctree::
   :maxdepth: 2

   shallow-fusion
   LODR
   rescoring
