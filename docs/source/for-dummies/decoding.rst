.. _dummies_tutorial_decoding:

Decoding
========

After :ref:`dummies_tutorial_training`, we can start decoding.

The command to start the decoding is quite simple:

.. code-block:: bash

   cd /tmp/icefall
   export PYTHONPATH=/tmp/icefall:$PYTHONPATH
   cd egs/yesno/ASR

   # We use CPU for decoding by setting the following environment variable
   export CUDA_VISIBLE_DEVICES=""

   ./tdnn/decode.py

The output logs are given below:

.. literalinclude:: ./code/decoding-yesno.txt

For the more curious
--------------------

.. code-block:: bash

   ./tdnn/decode.py --help

will print the usage information about ``./tdnn/decode.py``. For instance, you
can specify:

  - ``--epoch`` to use which checkpoint for decoding
  - ``--avg`` to select how many checkpoints to use for model averaging

You usually try different combinations of ``--epoch`` and ``--avg`` and select
one that leads to the lowest WER (`Word Error Rate <https://en.wikipedia.org/wiki/Word_error_rate>`_).
