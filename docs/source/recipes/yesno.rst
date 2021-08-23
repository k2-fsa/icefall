yesno
=====

This page shows you how to run the ``yesno`` recipe.

.. HINT::

  We assume you have read the page :ref:`install icefall` and have setup
  the environment for ``icefall``.

.. HINT::

  You **don't** need a **GPU** to run this recipe. It can be run on a **CPU**.
  The training time takes less than 30 **seconds** and you will get
  the following WER::

    [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]

Data preparation
----------------

.. code-block::

  $ cd egs/yesno/ASR
  $ ./prepare.sh

The script ``./prepare.sh`` handles the data preparation for you, automagically.
All you need to do is to run it.

The data preparation contains several stages, you can use the following two
options:

  - ``--stage``
  - ``--stop-stage``

to control which stage(s) should be run. By default, all stages are executed.


For example,

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ ./prepare.sh --stage 0 --stop-stage 0

means to run only stage 0.

To run stage 2 to stage 5, use:

.. code-block:: bash

  $ ./prepare.sh --stage 2 --stop-stage 5


Training
--------

We provide only a TDNN model, contained in
the `tdnn <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn>`_
folder, for ``yesno``.

The command to run the training part is:

.. code-block:: bash

  $ cd egs/yesno/ASR
  $ ./tdnn/train.py

By default, it will run ``15`` epochs. Training logs and checkpoints are saved
in ``tdnn/exp``.

To see the training options, you can use:

.. code-block:: bash

  $ ./tdnn/train.py --help

Decoding
--------

The decoding part uses checkpoints saved by the training part, so you have
to run the training part first.

The command for decoding is:

.. code-block:: bash

  $ ./tdnn/decode.py

You will see the WER in the output log.
Decoding results are saved in ``tdnn/exp``.

Colab notebook
--------------

We do provide a colab notebook for this recipe.

|yesno colab notebook|

.. |yesno colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing



Use a pre-trained model
-----------------------

TODO
