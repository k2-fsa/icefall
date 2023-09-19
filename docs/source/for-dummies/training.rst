.. _dummies_tutorial_training:

Training
========

After :ref:`dummies_tutorial_data_preparation`, we can start training.

The command to start the training is quite simple:

.. code-block:: bash

   cd /tmp/icefall
   export PYTHONPATH=/tmp/icefall:$PYTHONPATH
   cd egs/yesno/ASR

   # We use CPU for training by setting the following environment variable
   export CUDA_VISIBLE_DEVICES=""

   ./tdnn/train.py

That's it!

You can find the training logs below:

.. literalinclude:: ./code/train-yesno.txt

For the more curious
--------------------

.. code-block:: bash

   ./tdnn/train.py --help

will print the usage information about ``./tdnn/train.py``. For instance, you
can specify the number of epochs to train and the location to save the training
results.

The training text logs are saved in ``tdnn/exp/log`` while the tensorboard
logs are in ``tdnn/exp/tensorboard``.
