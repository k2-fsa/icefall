.. _dummies_tutorial_environment_setup:

Environment setup
=================

We will create an environment for `Next-gen Kaldi`_ that runs on ``CPU``
in this tutorial.

.. note::

   Since the `yesno`_ dataset used in this tutorial is very tiny, training on
   ``CPU`` works very well for it.

   If your dataset is very large, e.g., hundreds or thousands of hours of
   training data, please follow :ref:`install icefall` to install `icefall`_
   that works with ``GPU``.


Create a virtual environment
----------------------------

.. code-block:: bash

  virtualenv -p python3 /tmp/icefall_env

The above command creates a virtual environment in the directory ``/tmp/icefall_env``.
You can select any directory you want.

The output of the above command is given below:

.. code-block:: bash

  Already using interpreter /usr/bin/python3
  Using base prefix '/usr'
  New python executable in /tmp/icefall_env/bin/python3
  Also creating executable in /tmp/icefall_env/bin/python
  Installing setuptools, pkg_resources, pip, wheel...done.

Now we can activate the environment using:

.. code-block:: bash

  source /tmp/icefall_env/bin/activate

Install dependencies
--------------------

.. warning::

   Remeber to activate your virtual environment before you continue!

After activating the virtual environment, we can use the following command
to install dependencies of `icefall`_:

.. hint::

   Remeber that we will run this tutorial on ``CPU``, so we install
   dependencies required only by running on ``CPU``.

.. code-block:: bash

   # Caution: Installation order matters!

   # We use torch 2.0.0 and torchaduio 2.0.0 in this tutorial.
   # Other versions should also work.

   pip install torch==2.0.0+cpu torchaudio==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

   # If you are using macOS, please use the following command to install torch and torchaudio
   # pip install torch==2.0.0 torchaudio==2.0.0 -f https://download.pytorch.org/whl/torch_stable.html

   # Now install k2
   # Please refer to https://k2-fsa.github.io/k2/installation/from_wheels.html#linux-cpu-example

   pip install k2==1.24.4.dev20231220+cpu.torch2.0.0 -f https://k2-fsa.github.io/k2/cpu.html

   # For users from China
   # 中国国内用户，如果访问不了 huggingface, 请使用
   # pip install k2==1.24.4.dev20231220+cpu.torch2.0.0 -f https://k2-fsa.github.io/k2/cpu-cn.html

   # Install the latest version of lhotse

   pip install git+https://github.com/lhotse-speech/lhotse


Install icefall
---------------

We will put the source code of `icefall`_ into the directory ``/tmp``
You can select any directory you want.

.. code-block:: bash

   cd /tmp
   git clone https://github.com/k2-fsa/icefall
   cd icefall
   pip install -r ./requirements.txt

.. code-block:: bash

   # Anytime we want to use icefall, we have to set the following
   # environment variable

   export PYTHONPATH=/tmp/icefall:$PYTHONPATH

.. hint::

   If you get the following error during this tutorial:

    .. code-block:: bash

      ModuleNotFoundError: No module named 'icefall'

  please set the above environment variable to fix it.


Congratulations! You have installed `icefall`_ successfully.

For the more curious
--------------------

`icefall`_ contains a collection of Python scripts and you don't need to
use ``python3 setup.py install`` or ``pip install icefall`` to install it.
All you need to do is to download the code and set the environment variable
``PYTHONPATH``.
