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
