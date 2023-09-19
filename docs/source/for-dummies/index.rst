Icefall for dummies tutorial
============================

This tutorial walks you step by step about how to create a simple
ASR (`Automatic Speech Recognition <https://en.wikipedia.org/wiki/Speech_recognition>`_)
system with `Next-gen Kaldi`_.

We use the `yesno`_ dataset for demonstration. We select it out of two reasons:

  - It is quite tiny, containing only about 12 minutes of data
  - The training can be finished within 20 seconds on ``CPU``.

That also means you don't need a ``GPU`` to run this tutorial.

Let's get started!

Please follow items below **sequentially**.

.. note::

   The :ref:`dummies_tutorial_data_preparation` runs only on Linux and on macOS.
   All other parts run on Linux, macOS, and Windows.

   Help from the community is appreciated to port the :ref:`dummies_tutorial_data_preparation`
   to Windows.

.. toctree::
   :maxdepth: 2

   ./environment-setup.rst
   ./data-preparation.rst
   ./training.rst
   ./decoding.rst
   ./model-export.rst
