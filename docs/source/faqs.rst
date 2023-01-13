Frequently Asked Questions (FAQs)
=================================

In this section, we collect issues reported by users and post the corresponding
solutions.


OSError: libtorch_hip.so: cannot open shared object file: no such file or directory
-----------------------------------------------------------------------------------

One user is using the following code to install ``torch`` and ``torchaudio``:

.. code-block:: bash

  pip install \
    torch==1.10.0+cu111 \
    torchvision==0.11.0+cu111 \
    torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

and it throws the following error when running ``tdnn/train.py``:

.. code-block::

  OSError: libtorch_hip.so: cannot open shared object file: no such file or directory

The fix is to specify the CUDA version while installing ``torchaudio``. That
is, change ``torchaudio==0.10.0`` to ``torchaudio==0.10.0+cu11```. Therefore,
the correct command is:

.. code-block:: bash

  pip install \
    torch==1.10.0+cu111 \
    torchvision==0.11.0+cu111 \
    torchaudio==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

AttributeError: module 'distutils' has no attribute 'version'
-------------------------------------------------------------

The error log is:

.. code-block::

  Traceback (most recent call last):
    File "./tdnn/train.py", line 14, in <module>
      from asr_datamodule import YesNoAsrDataModule
    File "/home/xxx/code/next-gen-kaldi/icefall/egs/yesno/ASR/tdnn/asr_datamodule.py", line 34, in <module>
      from icefall.dataset.datamodule import DataModule
    File "/home/xxx/code/next-gen-kaldi/icefall/icefall/__init__.py", line 3, in <module>
      from . import (
    File "/home/xxx/code/next-gen-kaldi/icefall/icefall/decode.py", line 23, in <module>
      from icefall.utils import add_eos, add_sos, get_texts
    File "/home/xxx/code/next-gen-kaldi/icefall/icefall/utils.py", line 39, in <module>
      from torch.utils.tensorboard import SummaryWriter
    File "/home/xxx/tool/miniconda3/envs/yyy/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
      LooseVersion = distutils.version.LooseVersion
  AttributeError: module 'distutils' has no attribute 'version'

The fix is:

.. code-block:: bash

  pip uninstall setuptools

  pip install setuptools==58.0.4

ImportError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory
--------------------------------------------------------------------------------------------

If you are using ``conda`` and encounter the following issue:

.. code-block::

  Traceback (most recent call last):
    File "/k2-dev/yangyifan/anaconda3/envs/icefall/lib/python3.10/site-packages/k2-1.23.3.dev20230112+cuda11.6.torch1.13.1-py3.10-linux-x86_64.egg/k2/__init__.py", line 24, in <module>
      from _k2 import DeterminizeWeightPushingType
  ImportError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory

  During handling of the above exception, another exception occurred:

  Traceback (most recent call last):
    File "/k2-dev/yangyifan/icefall/egs/librispeech/ASR/./pruned_transducer_stateless7_ctc_bs/decode.py", line 104, in <module>
      import k2
    File "/k2-dev/yangyifan/anaconda3/envs/icefall/lib/python3.10/site-packages/k2-1.23.3.dev20230112+cuda11.6.torch1.13.1-py3.10-linux-x86_64.egg/k2/__init__.py", line 30, in <module>
      raise ImportError(
  ImportError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory
  Note: If you're using anaconda and importing k2 on MacOS,
        you can probably fix this by setting the environment variable:
    export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages:$DYLD_LIBRARY_PATH

Please first try to find where ``libpython3.10.so.1.0`` locates.

For instance,

.. code-block:: bash

  cd $CONDA_PREFIX/lib
  find . -name "libpython*"

If you are able to find it inside ``$CODNA_PREFIX/lib``, please set the
following environment variable:

.. code-block:: bash

  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
