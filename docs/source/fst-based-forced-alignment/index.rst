FST-based forced alignment
==========================

This section describes how to perform **FST-based** ``forced alignment`` with models
trained by the `CTC`_ loss.

We use `CTC FORCED ALIGNMENT API TUTORIAL <https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html>`_
from `torchaudio`_ as a reference in this section. The difference is that we are using an ``FST``-based approach.

Two approaches for FST-based forced alignment will be described:

  - `Kaldi`_-based
  - `k2`_-base

Note that the `Kaldi`_-based approach does not depend on `Kaldi`_ at all.
That is, you don't need to install `Kaldi`_ in order to use it. Instead,
we will use `kaldi-decoder`_, which has ported the C++ decoding code from `Kaldi`_
without depending on it.


Differences between the two approaches
--------------------------------------

The following table compares the differences between the two approaches.

.. list-table::

 * - Features
   - `Kaldi`_-based
   - `k2`_-based
 * - Support CUDA
   - No
   - Yes
 * - Support CPU
   - Yes
   - Yes
 * - Support batch processing
   - No
   - Yes on CUDA; No on CPU
 * - Support streaming models
   - Yes
   - No
 * - Support C++ APIs
   - Yes
   - Yes
 * - Support Python APIs
   - Yes
   - Yes


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   kaldi-based
   k2-based
