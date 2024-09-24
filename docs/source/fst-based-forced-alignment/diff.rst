Two approaches
==============

Two approaches for FST-based forced alignment will be described:

  - `Kaldi`_-based
  - `k2`_-based

Note that the `Kaldi`_-based approach does not depend on `Kaldi`_ at all.
That is, you don't need to install `Kaldi`_ in order to use it. Instead,
we use `kaldi-decoder`_, which has ported the C++ decoding code from `Kaldi`_
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
