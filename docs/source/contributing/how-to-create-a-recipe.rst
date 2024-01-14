How to create a recipe
======================

.. HINT::

  Please read :ref:`follow the code style` to adjust your code style.

.. CAUTION::

  ``icefall`` is designed to be as Pythonic as possible. Please use
  Python in your recipe if possible.

Data Preparation
----------------

We recommend you to prepare your training/test/validate dataset
with `lhotse <https://github.com/lhotse-speech/lhotse>`_.

Please refer to `<https://lhotse.readthedocs.io/en/latest/index.html>`_
for how to create a recipe in ``lhotse``.

.. HINT::

  The ``yesno`` recipe in ``lhotse`` is a very good example.

  Please refer to `<https://github.com/lhotse-speech/lhotse/pull/380>`_,
  which shows how to add a new recipe to ``lhotse``.

Suppose you would like to add a recipe for a dataset named ``foo``.
You can do the following:

.. code-block::

  $ cd egs
  $ mkdir -p foo/ASR
  $ cd foo/ASR
  $ touch prepare.sh
  $ chmod +x prepare.sh

If your dataset is very simple, please follow
`egs/yesno/ASR/prepare.sh <https://github.com/k2-fsa/icefall/blob/master/egs/yesno/ASR/prepare.sh>`_
to write your own ``prepare.sh``.
Otherwise, please refer to
`egs/librispeech/ASR/prepare.sh <https://github.com/k2-fsa/icefall/blob/master/egs/yesno/ASR/prepare.sh>`_
to prepare your data.


Training
--------

Assume you have a fancy model, called ``bar`` for the ``foo`` recipe, you can
organize your files in the following way:

.. code-block::

  $ cd egs/foo/ASR
  $ mkdir bar
  $ cd bar
  $ touch README.md model.py train.py decode.py asr_datamodule.py pretrained.py

For instance , the ``yesno`` recipe has a ``tdnn`` model and its directory structure
looks like the following:

.. code-block:: bash

  egs/yesno/ASR/tdnn/
  |-- README.md
  |-- asr_datamodule.py
  |-- decode.py
  |-- model.py
  |-- pretrained.py
  `-- train.py

**File description**:

  - ``README.md``

    It contains information of this recipe, e.g., how to run it, what the WER is, etc.

  - ``asr_datamodule.py``

    It provides code to create PyTorch dataloaders with train/test/validation dataset.

  - ``decode.py``

    It takes as inputs the checkpoints saved during the training stage to decode the test
    dataset(s).

  - ``model.py``

    It contains the definition of your fancy neural network model.

  - ``pretrained.py``

    We can use this script to do inference with a pre-trained model.

  - ``train.py``

    It contains training code.


.. HINT::

  Please take a look at

    - `egs/yesno/tdnn <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn>`_
    - `egs/librispeech/tdnn_lstm_ctc <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/tdnn_lstm_ctc>`_
    - `egs/librispeech/conformer_ctc <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conformer_ctc>`_

  to get a feel what the resulting files look like.

.. NOTE::

  Every model in a recipe is kept to be as self-contained as possible.
  We tolerate duplicate code among different recipes.


The training stage should be invocable by:

  .. code-block::

    $ cd egs/foo/ASR
    $ ./bar/train.py
    $ ./bar/train.py --help


Decoding
--------

Please refer to

  - `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/decode.py>`_

    If your model is transformer/conformer based.

  - `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/tdnn_lstm_ctc/decode.py>`_

    If your model is TDNN/LSTM based, i.e., there is no attention decoder.

  - `<https://github.com/k2-fsa/icefall/blob/master/egs/yesno/ASR/tdnn/decode.py>`_

    If there is no LM rescoring.

The decoding stage should be invocable by:

  .. code-block::

    $ cd egs/foo/ASR
    $ ./bar/decode.py
    $ ./bar/decode.py --help

Pre-trained model
-----------------

Please demonstrate how to use your model for inference in ``egs/foo/ASR/bar/pretrained.py``.
If possible, please consider creating a Colab notebook to show that.
