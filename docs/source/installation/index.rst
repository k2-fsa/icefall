.. _install icefall:

Installation
============

.. hint::

   We also provide :ref:`icefall_docker` support, which has already setup
   the environment for you.

.. hint::

  We have a colab notebook guiding you step by step to setup the environment.

  |yesno colab notebook|

  .. |yesno colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing

`icefall`_ depends on `k2`_ and `lhotse`_.

We recommend that you use the following steps to install the dependencies.

- (0) Install CUDA toolkit and cuDNN
- (1) Install `torch`_ and `torchaudio`_
- (2) Install `k2`_
- (3) Install `lhotse`_

.. caution::

  Installation order matters.

(0) Install CUDA toolkit and cuDNN
----------------------------------

Please refer to
`<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
to install CUDA and cuDNN.


(1) Install torch and torchaudio
--------------------------------

Please refer `<https://pytorch.org/>`_ to install `torch`_ and `torchaudio`_.

.. caution::

   Please install torch and torchaudio at the same time.

(2) Install k2
--------------

Please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install `k2`_.

.. caution::

  Please don't change your installed PyTorch after you have installed k2.

.. note::

   We suggest that you install k2 from pre-compiled wheels by following
   `<https://k2-fsa.github.io/k2/installation/from_wheels.html>`_

.. hint::

   Please always install the latest version of `k2`_.

(3) Install lhotse
------------------

Please refer to `<https://lhotse.readthedocs.io/en/latest/getting-started.html#installation>`_
to install `lhotse`_.

.. hint::

    We strongly recommend you to use::

      pip install git+https://github.com/lhotse-speech/lhotse

    to install the latest version of `lhotse`_.

(4) Download icefall
--------------------

`icefall`_ is a collection of Python scripts; what you need is to download it
and set the environment variable ``PYTHONPATH`` to point to it.

Assume you want to place `icefall`_ in the folder ``/tmp``. The
following commands show you how to setup `icefall`_:

.. code-block:: bash

  cd /tmp
  git clone https://github.com/k2-fsa/icefall
  cd icefall
  pip install -r requirements.txt
  export PYTHONPATH=/tmp/icefall:$PYTHONPATH

.. HINT::

  You can put several versions of `icefall`_ in the same virtual environment.
  To switch among different versions of `icefall`_, just set ``PYTHONPATH``
  to point to the version you want.

Installation example
--------------------

The following shows an example about setting up the environment.

(1) Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   kuangfangjun:~$ virtualenv -p python3.8 test-icefall
   created virtual environment CPython3.8.0.final.0-64 in 9422ms
     creator CPython3Posix(dest=/star-fj/fangjun/test-icefall, clear=False, no_vcs_ignore=False, global=False)
     seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/star-fj/fangjun/.local/share/virtualenv)
       added seed packages: pip==22.3.1, setuptools==65.6.3, wheel==0.38.4
     activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

   kuangfangjun:~$ source test-icefall/bin/activate

   (test-icefall) kuangfangjun:~$

(2) Install CUDA toolkit and cuDNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to determine the version of CUDA toolkit to install.

.. code-block:: bash

   (test-icefall) kuangfangjun:~$ nvidia-smi | head -n 4

   Wed Jul 26 21:57:49 2023
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
   |-------------------------------+----------------------+----------------------+

You can choose any CUDA version that is ``not`` greater than the version printed by ``nvidia-smi``.
In our case, we can choose any version ``<= 11.6``.

We will use ``CUDA 11.6`` in this example. Please follow
`<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html#cuda-11-6>`_
to install CUDA toolkit and cuDNN if you have not done that before.

After installing CUDA toolkit, you can use the following command to verify it:

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ nvcc --version

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2019 NVIDIA Corporation
  Built on Wed_Oct_23_19:24:38_PDT_2019
  Cuda compilation tools, release 10.2, V10.2.89

(3) Install torch and torchaudio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since we have selected CUDA toolkit ``11.6``, we have to install a version of `torch`_
that is compiled against CUDA ``11.6``. We select ``torch 1.13.0+cu116`` in this
example.

After selecting the version of `torch`_ to install, we need to also install
a compatible version of `torchaudio`_, which is ``0.13.0+cu116`` in our case.

Please refer to `<https://pytorch.org/audio/stable/installation.html#compatibility-matrix>`_
to select an appropriate version of `torchaudio`_ to install if you use a different
version of `torch`_.

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ pip install torch==1.13.0+cu116 torchaudio==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

  Looking in links: https://download.pytorch.org/whl/torch_stable.html
  Collecting torch==1.13.0+cu116
    Downloading https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl (1983.0 MB)
       ________________________________________ 2.0/2.0 GB 764.4 kB/s eta 0:00:00
  Collecting torchaudio==0.13.0+cu116
    Downloading https://download.pytorch.org/whl/cu116/torchaudio-0.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl (4.2 MB)
       ________________________________________ 4.2/4.2 MB 1.3 MB/s eta 0:00:00
  Requirement already satisfied: typing-extensions in /star-fj/fangjun/test-icefall/lib/python3.8/site-packages (from torch==1.13.0+cu116) (4.7.1)
  Installing collected packages: torch, torchaudio
  Successfully installed torch-1.13.0+cu116 torchaudio-0.13.0+cu116

Verify that `torch`_ and `torchaudio`_ are successfully installed:

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ python3 -c "import torch; print(torch.__version__)"

  1.13.0+cu116

  (test-icefall) kuangfangjun:~$ python3 -c "import torchaudio; print(torchaudio.__version__)"

  0.13.0+cu116

(4) Install k2
~~~~~~~~~~~~~~

We will install `k2`_ from pre-compiled wheels by following
`<https://k2-fsa.github.io/k2/installation/from_wheels.html>`_

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ pip install k2==1.24.3.dev20230725+cuda11.6.torch1.13.0 -f https://k2-fsa.github.io/k2/cuda.html

  Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
  Looking in links: https://k2-fsa.github.io/k2/cuda.html
  Collecting k2==1.24.3.dev20230725+cuda11.6.torch1.13.0
    Downloading https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.3.dev20230725%2Bcuda11.6.torch1.13.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (104.3 MB)
       ________________________________________ 104.3/104.3 MB 5.1 MB/s eta 0:00:00
  Requirement already satisfied: torch==1.13.0 in /star-fj/fangjun/test-icefall/lib/python3.8/site-packages (from k2==1.24.3.dev20230725+cuda11.6.torch1.13.0) (1.13.0+cu116)
  Collecting graphviz
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/de/5e/fcbb22c68208d39edff467809d06c9d81d7d27426460ebc598e55130c1aa/graphviz-0.20.1-py3-none-any.whl (47 kB)
  Requirement already satisfied: typing-extensions in /star-fj/fangjun/test-icefall/lib/python3.8/site-packages (from torch==1.13.0->k2==1.24.3.dev20230725+cuda11.6.torch1.13.0) (4.7.1)
  Installing collected packages: graphviz, k2
  Successfully installed graphviz-0.20.1 k2-1.24.3.dev20230725+cuda11.6.torch1.13.0

.. hint::

   Please refer to `<https://k2-fsa.github.io/k2/cuda.html>`_ for the available
   pre-compiled wheels about `k2`_.

Verify that `k2`_ has been installed successfully:

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ python3 -m k2.version

  Collecting environment information...

  k2 version: 1.24.3
  Build type: Release
  Git SHA1: 4c05309499a08454997adf500b56dcc629e35ae5
  Git date: Tue Jul 25 16:23:36 2023
  Cuda used to build k2: 11.6
  cuDNN used to build k2: 8.3.2
  Python version used to build k2: 3.8
  OS used to build k2: CentOS Linux release 7.9.2009 (Core)
  CMake version: 3.27.0
  GCC version: 9.3.1
  CMAKE_CUDA_FLAGS:  -Wno-deprecated-gpu-targets   -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_35,code=sm_35  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_50,code=sm_50  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_60,code=sm_60  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_61,code=sm_61  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_70,code=sm_70  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_75,code=sm_75  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_80,code=sm_80  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_86,code=sm_86 -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=integer_sign_change,--diag_suppress=useless_using_declaration,--diag_suppress=set_but_not_used,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=unsigned_compare_with_zero,--diag_suppress=declared_but_not_referenced,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -D_GLIBCXX_USE_CXX11_ABI=0 --compiler-options -Wall  --compiler-options -Wno-strict-overflow  --compiler-options -Wno-unknown-pragmas
  CMAKE_CXX_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-unused-variable  -Wno-strict-overflow
  PyTorch version used to build k2: 1.13.0+cu116
  PyTorch is using Cuda: 11.6
  NVTX enabled: True
  With CUDA: True
  Disable debug: True
  Sync kernels : False
  Disable checks: False
  Max cpu memory allocate: 214748364800 bytes (or 200.0 GB)
  k2 abort: False
  __file__: /star-fj/fangjun/test-icefall/lib/python3.8/site-packages/k2/version/version.py
  _k2.__file__: /star-fj/fangjun/test-icefall/lib/python3.8/site-packages/_k2.cpython-38-x86_64-linux-gnu.so

(5) Install lhotse
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ pip install git+https://github.com/lhotse-speech/lhotse

  Collecting git+https://github.com/lhotse-speech/lhotse
    Cloning https://github.com/lhotse-speech/lhotse to /tmp/pip-req-build-vq12fd5i
    Running command git clone --filter=blob:none --quiet https://github.com/lhotse-speech/lhotse /tmp/pip-req-build-vq12fd5i
    Resolved https://github.com/lhotse-speech/lhotse to commit 7640d663469b22cd0b36f3246ee9b849cd25e3b7
    Installing build dependencies ... done
    Getting requirements to build wheel ... done
    Preparing metadata (pyproject.toml) ... done
  Collecting cytoolz>=0.10.1
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1e/3b/a7828d575aa17fb7acaf1ced49a3655aa36dad7e16eb7e6a2e4df0dda76f/cytoolz-0.12.2-cp38-cp38-
  manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
       ________________________________________ 2.0/2.0 MB 33.2 MB/s eta 0:00:00
  Collecting pyyaml>=5.3.1
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c8/6b/6600ac24725c7388255b2f5add93f91e58a5d7efaf4af244fdbcc11a541b/PyYAML-6.0.1-cp38-cp38-ma
  nylinux_2_17_x86_64.manylinux2014_x86_64.whl (736 kB)
       ________________________________________ 736.6/736.6 kB 38.6 MB/s eta 0:00:00
  Collecting dataclasses
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/26/2f/1095cdc2868052dd1e64520f7c0d5c8c550ad297e944e641dbf1ffbb9a5d/dataclasses-0.6-py3-none-
  any.whl (14 kB)
  Requirement already satisfied: torchaudio in ./test-icefall/lib/python3.8/site-packages (from lhotse==1.16.0.dev0+git.7640d66.clean) (0.13.0+cu116)
  Collecting lilcom>=1.1.0
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a8/65/df0a69c52bd085ca1ad4e5c4c1a5c680e25f9477d8e49316c4ff1e5084a4/lilcom-1.7-cp38-cp38-many
  linux_2_17_x86_64.manylinux2014_x86_64.whl (87 kB)
       ________________________________________ 87.1/87.1 kB 8.7 MB/s eta 0:00:00
  Collecting tqdm
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/e6/02/a2cff6306177ae6bc73bc0665065de51dfb3b9db7373e122e2735faf0d97/tqdm-4.65.0-py3-none-any
  .whl (77 kB)
  Requirement already satisfied: numpy>=1.18.1 in ./test-icefall/lib/python3.8/site-packages (from lhotse==1.16.0.dev0+git.7640d66.clean) (1.24.4)
  Collecting audioread>=2.1.9
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/5d/cb/82a002441902dccbe427406785db07af10182245ee639ea9f4d92907c923/audioread-3.0.0.tar.gz (
  377 kB)
    Preparing metadata (setup.py) ... done
  Collecting tabulate>=0.8.1
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/40/44/4a5f08c96eb108af5cb50b41f76142f0afa346dfa99d5296fe7202a11854/tabulate-0.9.0-py3-none-
  any.whl (35 kB)
  Collecting click>=7.1.1
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1a/70/e63223f8116931d365993d4a6b7ef653a4d920b41d03de7c59499962821f/click-8.1.6-py3-none-any.
  whl (97 kB)
       ________________________________________ 97.9/97.9 kB 8.4 MB/s eta 0:00:00
  Collecting packaging
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/ab/c3/57f0601a2d4fe15de7a553c00adbc901425661bf048f2a22dfc500caf121/packaging-23.1-py3-none-
  any.whl (48 kB)
  Collecting intervaltree>=3.1.0
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/50/fb/396d568039d21344639db96d940d40eb62befe704ef849b27949ded5c3bb/intervaltree-3.1.0.tar.gz
   (32 kB)
    Preparing metadata (setup.py) ... done
  Requirement already satisfied: torch in ./test-icefall/lib/python3.8/site-packages (from lhotse==1.16.0.dev0+git.7640d66.clean) (1.13.0+cu116)
  Collecting SoundFile>=0.10
    Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ad/bd/0602167a213d9184fc688b1086dc6d374b7ae8c33eccf169f9b50ce6568c/soundfile-0.12.1-py2.py3-
  none-manylinux_2_17_x86_64.whl (1.3 MB)
       ________________________________________ 1.3/1.3 MB 46.5 MB/s eta 0:00:00
  Collecting toolz>=0.8.0
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/7f/5c/922a3508f5bda2892be3df86c74f9cf1e01217c2b1f8a0ac4841d903e3e9/toolz-0.12.0-py3-none-any.whl (55 kB)
  Collecting sortedcontainers<3.0,>=2.0
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/32/46/9cb0e58b2deb7f82b84065f37f3bffeb12413f947f9388e4cac22c4621ce/sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
  Collecting cffi>=1.0
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/b7/8b/06f30caa03b5b3ac006de4f93478dbd0239e2a16566d81a106c322dc4f79/cffi-1.15.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (442 kB)
  Requirement already satisfied: typing-extensions in ./test-icefall/lib/python3.8/site-packages (from torch->lhotse==1.16.0.dev0+git.7640d66.clean) (4.7.1)
  Collecting pycparser
    Using cached https://pypi.tuna.tsinghua.edu.cn/packages/62/d5/5f610ebe421e85889f2e55e33b7f9a6795bd982198517d912eb1c76e1a53/pycparser-2.21-py2.py3-none-any.whl (118 kB)
  Building wheels for collected packages: lhotse, audioread, intervaltree
    Building wheel for lhotse (pyproject.toml) ... done
    Created wheel for lhotse: filename=lhotse-1.16.0.dev0+git.7640d66.clean-py3-none-any.whl size=687627 sha256=cbf0a4d2d0b639b33b91637a4175bc251d6a021a069644ecb1a9f2b3a83d072a
    Stored in directory: /tmp/pip-ephem-wheel-cache-wwtk90_m/wheels/7f/7a/8e/a0bf241336e2e3cb573e1e21e5600952d49f5162454f2e612f
    Building wheel for audioread (setup.py) ... done
    Created wheel for audioread: filename=audioread-3.0.0-py3-none-any.whl size=23704 sha256=5e2d3537c96ce9cf0f645a654c671163707bf8cb8d9e358d0e2b0939a85ff4c2
    Stored in directory: /star-fj/fangjun/.cache/pip/wheels/e2/c3/9c/f19ae5a03f8862d9f0776b0c0570f1fdd60a119d90954e3f39
    Building wheel for intervaltree (setup.py) ... done
    Created wheel for intervaltree: filename=intervaltree-3.1.0-py2.py3-none-any.whl size=26098 sha256=2604170976cfffe0d2f678cb1a6e5b525f561cd50babe53d631a186734fec9f9
    Stored in directory: /star-fj/fangjun/.cache/pip/wheels/f3/ed/2b/c179ebfad4e15452d6baef59737f27beb9bfb442e0620f7271
  Successfully built lhotse audioread intervaltree
  Installing collected packages: sortedcontainers, dataclasses, tqdm, toolz, tabulate, pyyaml, pycparser, packaging, lilcom, intervaltree, click, audioread, cytoolz, cffi, SoundFile, lhotse
  Successfully installed SoundFile-0.12.1 audioread-3.0.0 cffi-1.15.1 click-8.1.6 cytoolz-0.12.2 dataclasses-0.6 intervaltree-3.1.0 lhotse-1.16.0.dev0+git.7640d66.clean lilcom-1.7 packaging-23.1 pycparser-2.21 pyyaml-6.0.1 sortedcontainers-2.4.0 tabulate-0.9.0 toolz-0.12.0 tqdm-4.65.0


Verify that `lhotse`_ has been installed successfully:

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ python3 -c "import lhotse; print(lhotse.__version__)"

  1.16.0.dev+git.7640d66.clean

(6) Download icefall
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ cd /tmp/

  (test-icefall) kuangfangjun:tmp$ git clone https://github.com/k2-fsa/icefall

  Cloning into 'icefall'...
  remote: Enumerating objects: 12942, done.
  remote: Counting objects: 100% (67/67), done.
  remote: Compressing objects: 100% (56/56), done.
  remote: Total 12942 (delta 17), reused 35 (delta 6), pack-reused 12875
  Receiving objects: 100% (12942/12942), 14.77 MiB | 9.29 MiB/s, done.
  Resolving deltas: 100% (8835/8835), done.

  (test-icefall) kuangfangjun:tmp$ cd icefall/

  (test-icefall) kuangfangjun:icefall$ pip install -r ./requirements.txt

Test Your Installation
----------------------

To test that your installation is successful, let us run
the `yesno recipe <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR>`_
on ``CPU``.

Data preparation
~~~~~~~~~~~~~~~~

.. code-block:: bash

  (test-icefall) kuangfangjun:icefall$ export PYTHONPATH=/tmp/icefall:$PYTHONPATH

  (test-icefall) kuangfangjun:icefall$ cd /tmp/icefall

  (test-icefall) kuangfangjun:icefall$ cd egs/yesno/ASR

  (test-icefall) kuangfangjun:ASR$ ./prepare.sh


The log of running ``./prepare.sh`` is:

.. code-block::

  2023-07-27 12:41:39 (prepare.sh:27:main) dl_dir: /tmp/icefall/egs/yesno/ASR/download
  2023-07-27 12:41:39 (prepare.sh:30:main) Stage 0: Download data
  /tmp/icefall/egs/yesno/ASR/download/waves_yesno.tar.gz: 100%|___________________________________________________| 4.70M/4.70M [00:00<00:00, 11.1MB/s]
  2023-07-27 12:41:46 (prepare.sh:39:main) Stage 1: Prepare yesno manifest
  2023-07-27 12:41:50 (prepare.sh:45:main) Stage 2: Compute fbank for yesno
  2023-07-27 12:41:55,718 INFO [compute_fbank_yesno.py:65] Processing train
  Extracting and storing features: 100%|_______________________________________________________________________________| 90/90 [00:01<00:00, 87.82it/s]
  2023-07-27 12:41:56,778 INFO [compute_fbank_yesno.py:65] Processing test
  Extracting and storing features: 100%|______________________________________________________________________________| 30/30 [00:00<00:00, 256.92it/s]
  2023-07-27 12:41:57 (prepare.sh:51:main) Stage 3: Prepare lang
  2023-07-27 12:42:02 (prepare.sh:66:main) Stage 4: Prepare G
  /project/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Read(std::istream&):79
  [I] Reading \data\ section.
  /project/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Read(std::istream&):140
  [I] Reading \1-grams: section.
  2023-07-27 12:42:02 (prepare.sh:92:main) Stage 5: Compile HLG
  2023-07-27 12:42:07,275 INFO [compile_hlg.py:124] Processing data/lang_phone
  2023-07-27 12:42:07,276 INFO [lexicon.py:171] Converting L.pt to Linv.pt
  2023-07-27 12:42:07,309 INFO [compile_hlg.py:48] Building ctc_topo. max_token_id: 3
  2023-07-27 12:42:07,310 INFO [compile_hlg.py:52] Loading G.fst.txt
  2023-07-27 12:42:07,314 INFO [compile_hlg.py:62] Intersecting L and G
  2023-07-27 12:42:07,323 INFO [compile_hlg.py:64] LG shape: (4, None)
  2023-07-27 12:42:07,323 INFO [compile_hlg.py:66] Connecting LG
  2023-07-27 12:42:07,323 INFO [compile_hlg.py:68] LG shape after k2.connect: (4, None)
  2023-07-27 12:42:07,323 INFO [compile_hlg.py:70] <class 'torch.Tensor'>
  2023-07-27 12:42:07,323 INFO [compile_hlg.py:71] Determinizing LG
  2023-07-27 12:42:07,341 INFO [compile_hlg.py:74] <class '_k2.ragged.RaggedTensor'>
  2023-07-27 12:42:07,341 INFO [compile_hlg.py:76] Connecting LG after k2.determinize
  2023-07-27 12:42:07,341 INFO [compile_hlg.py:79] Removing disambiguation symbols on LG
  2023-07-27 12:42:07,354 INFO [compile_hlg.py:91] LG shape after k2.remove_epsilon: (6, None)
  2023-07-27 12:42:07,445 INFO [compile_hlg.py:96] Arc sorting LG
  2023-07-27 12:42:07,445 INFO [compile_hlg.py:99] Composing H and LG
  2023-07-27 12:42:07,446 INFO [compile_hlg.py:106] Connecting LG
  2023-07-27 12:42:07,446 INFO [compile_hlg.py:109] Arc sorting LG
  2023-07-27 12:42:07,447 INFO [compile_hlg.py:111] HLG.shape: (8, None)
  2023-07-27 12:42:07,447 INFO [compile_hlg.py:127] Saving HLG.pt to data/lang_phone

Training
~~~~~~~~

Now let us run the training part:

.. code-block::

  (test-icefall) kuangfangjun:ASR$ export CUDA_VISIBLE_DEVICES=""

  (test-icefall) kuangfangjun:ASR$ ./tdnn/train.py

.. CAUTION::

  We use ``export CUDA_VISIBLE_DEVICES=""`` so that `icefall`_ uses CPU
  even if there are GPUs available.

.. hint::

   In case you get a ``Segmentation fault (core dump)`` error, please use:

      .. code-block:: bash

        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

   See more at `<https://github.com/k2-fsa/icefall/issues/674>` if you are
   interested.

The training log is given below:

.. code-block::

    2023-07-27 12:50:51,936 INFO [train.py:481] Training started
    2023-07-27 12:50:51,936 INFO [train.py:482] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01, 'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, 'best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'reset_interval': 20, 'valid_interval': 10, 'beam_size': 10, 'reduction': 'sum', 'use_double_scores': True, 'world_size': 1, 'master_port': 12354, 'tensorboard': True, 'num_epochs': 15, 'seed': 42, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 'num_workers': 2, 'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '4c05309499a08454997adf500b56dcc629e35ae5', 'k2-git-date': 'Tue Jul 25 16:23:36 2023', 'lhotse-version': '1.16.0.dev+git.7640d66.clean', 'torch-version': '1.13.0+cu116', 'torch-cuda-available': False, 'torch-cuda-version': '11.6', 'python-version': '3.8', 'icefall-git-branch': 'master', 'icefall-git-sha1': '3fb0a43-clean', 'icefall-git-date': 'Thu Jul 27 12:36:05 2023', 'icefall-path': '/tmp/icefall', 'k2-path': '/star-fj/fangjun/test-icefall/lib/python3.8/site-packages/k2/__init__.py', 'lhotse-path': '/star-fj/fangjun/test-icefall/lib/python3.8/site-packages/lhotse/__init__.py', 'hostname': 'de-74279-k2-train-1-1220091118-57c4d55446-sph26', 'IP address': '10.177.77.20'}}
    2023-07-27 12:50:51,941 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
    2023-07-27 12:50:51,949 INFO [train.py:495] device: cpu
    2023-07-27 12:50:51,965 INFO [asr_datamodule.py:146] About to get train cuts
    2023-07-27 12:50:51,965 INFO [asr_datamodule.py:244] About to get train cuts
    2023-07-27 12:50:51,967 INFO [asr_datamodule.py:149] About to create train dataset
    2023-07-27 12:50:51,967 INFO [asr_datamodule.py:199] Using SingleCutSampler.
    2023-07-27 12:50:51,967 INFO [asr_datamodule.py:205] About to create train dataloader
    2023-07-27 12:50:51,968 INFO [asr_datamodule.py:218] About to get test cuts
    2023-07-27 12:50:51,968 INFO [asr_datamodule.py:252] About to get test cuts
    2023-07-27 12:50:52,565 INFO [train.py:422] Epoch 0, batch 0, loss[loss=1.065, over 2436.00 frames. ], tot_loss[loss=1.065, over 2436.00 frames. ], batch size: 4
    2023-07-27 12:50:53,681 INFO [train.py:422] Epoch 0, batch 10, loss[loss=0.4561, over 2828.00 frames. ], tot_loss[loss=0.7076, over 22192.90 frames.], batch size: 4
    2023-07-27 12:50:54,167 INFO [train.py:444] Epoch 0, validation loss=0.9002, over 18067.00 frames.
    2023-07-27 12:50:55,011 INFO [train.py:422] Epoch 0, batch 20, loss[loss=0.2555, over 2695.00 frames. ], tot_loss[loss=0.484, over 34971.47 frames. ], batch size: 5
    2023-07-27 12:50:55,331 INFO [train.py:444] Epoch 0, validation loss=0.4688, over 18067.00 frames.
    2023-07-27 12:50:55,368 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-0.pt
    2023-07-27 12:50:55,633 INFO [train.py:422] Epoch 1, batch 0, loss[loss=0.2532, over 2436.00 frames. ], tot_loss[loss=0.2532, over 2436.00 frames. ],
     batch size: 4
    2023-07-27 12:50:56,242 INFO [train.py:422] Epoch 1, batch 10, loss[loss=0.1139, over 2828.00 frames. ], tot_loss[loss=0.1592, over 22192.90 frames.], batch size: 4
    2023-07-27 12:50:56,522 INFO [train.py:444] Epoch 1, validation loss=0.1627, over 18067.00 frames.
    2023-07-27 12:50:57,209 INFO [train.py:422] Epoch 1, batch 20, loss[loss=0.07055, over 2695.00 frames. ], tot_loss[loss=0.1175, over 34971.47 frames.], batch size: 5
    2023-07-27 12:50:57,600 INFO [train.py:444] Epoch 1, validation loss=0.07091, over 18067.00 frames.
    2023-07-27 12:50:57,640 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-1.pt
    2023-07-27 12:50:57,847 INFO [train.py:422] Epoch 2, batch 0, loss[loss=0.07731, over 2436.00 frames. ], tot_loss[loss=0.07731, over 2436.00 frames.], batch size: 4
    2023-07-27 12:50:58,427 INFO [train.py:422] Epoch 2, batch 10, loss[loss=0.04391, over 2828.00 frames. ], tot_loss[loss=0.05341, over 22192.90 frames. ], batch size: 4
    2023-07-27 12:50:58,884 INFO [train.py:444] Epoch 2, validation loss=0.04384, over 18067.00 frames.
    2023-07-27 12:50:59,387 INFO [train.py:422] Epoch 2, batch 20, loss[loss=0.03458, over 2695.00 frames. ], tot_loss[loss=0.04616, over 34971.47 frames. ], batch size: 5
    2023-07-27 12:50:59,707 INFO [train.py:444] Epoch 2, validation loss=0.03379, over 18067.00 frames.
    2023-07-27 12:50:59,758 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-2.pt

      ... ...

    2023-07-27 12:51:23,433 INFO [train.py:422] Epoch 13, batch 0, loss[loss=0.01054, over 2436.00 frames. ], tot_loss[loss=0.01054, over 2436.00 frames. ], batch size: 4
    2023-07-27 12:51:23,980 INFO [train.py:422] Epoch 13, batch 10, loss[loss=0.009014, over 2828.00 frames. ], tot_loss[loss=0.009974, over 22192.90 frames. ], batch size: 4
    2023-07-27 12:51:24,489 INFO [train.py:444] Epoch 13, validation loss=0.01085, over 18067.00 frames.
    2023-07-27 12:51:25,258 INFO [train.py:422] Epoch 13, batch 20, loss[loss=0.01172, over 2695.00 frames. ], tot_loss[loss=0.01055, over 34971.47 frames. ], batch size: 5
    2023-07-27 12:51:25,621 INFO [train.py:444] Epoch 13, validation loss=0.01074, over 18067.00 frames.
    2023-07-27 12:51:25,699 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-13.pt
    2023-07-27 12:51:25,866 INFO [train.py:422] Epoch 14, batch 0, loss[loss=0.01044, over 2436.00 frames. ], tot_loss[loss=0.01044, over 2436.00 frames. ], batch size: 4
    2023-07-27 12:51:26,844 INFO [train.py:422] Epoch 14, batch 10, loss[loss=0.008942, over 2828.00 frames. ], tot_loss[loss=0.01, over 22192.90 frames. ], batch size: 4
    2023-07-27 12:51:27,221 INFO [train.py:444] Epoch 14, validation loss=0.01082, over 18067.00 frames.
    2023-07-27 12:51:27,970 INFO [train.py:422] Epoch 14, batch 20, loss[loss=0.01169, over 2695.00 frames. ], tot_loss[loss=0.01054, over 34971.47 frames. ], batch size: 5
    2023-07-27 12:51:28,247 INFO [train.py:444] Epoch 14, validation loss=0.01073, over 18067.00 frames.
    2023-07-27 12:51:28,323 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-14.pt
    2023-07-27 12:51:28,326 INFO [train.py:555] Done!

Decoding
~~~~~~~~

Let us use the trained model to decode the test set:

.. code-block::

  (test-icefall) kuangfangjun:ASR$ ./tdnn/decode.py

  2023-07-27 12:55:12,840 INFO [decode.py:263] Decoding started
  2023-07-27 12:55:12,840 INFO [decode.py:264] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lm_dir': PosixPath('data/lm'), 'feature_dim': 23, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 14, 'avg': 2, 'export': False, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 'num_workers': 2, 'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '4c05309499a08454997adf500b56dcc629e35ae5', 'k2-git-date': 'Tue Jul 25 16:23:36 2023', 'lhotse-version': '1.16.0.dev+git.7640d66.clean', 'torch-version': '1.13.0+cu116', 'torch-cuda-available': False, 'torch-cuda-version': '11.6', 'python-version': '3.8', 'icefall-git-branch': 'master', 'icefall-git-sha1': '3fb0a43-clean', 'icefall-git-date': 'Thu Jul 27 12:36:05 2023', 'icefall-path': '/tmp/icefall', 'k2-path': '/star-fj/fangjun/test-icefall/lib/python3.8/site-packages/k2/__init__.py', 'lhotse-path': '/star-fj/fangjun/test-icefall/lib/python3.8/site-packages/lhotse/__init__.py', 'hostname': 'de-74279-k2-train-1-1220091118-57c4d55446-sph26', 'IP address': '10.177.77.20'}}
  2023-07-27 12:55:12,841 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
  2023-07-27 12:55:12,855 INFO [decode.py:273] device: cpu
  2023-07-27 12:55:12,868 INFO [decode.py:291] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
  2023-07-27 12:55:12,882 INFO [asr_datamodule.py:218] About to get test cuts
  2023-07-27 12:55:12,883 INFO [asr_datamodule.py:252] About to get test cuts
  2023-07-27 12:55:13,157 INFO [decode.py:204] batch 0/?, cuts processed until now is 4
  2023-07-27 12:55:13,701 INFO [decode.py:241] The transcripts are stored in tdnn/exp/recogs-test_set.txt
  2023-07-27 12:55:13,702 INFO [utils.py:564] [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
  2023-07-27 12:55:13,704 INFO [decode.py:249] Wrote detailed error stats to tdnn/exp/errs-test_set.txt
  2023-07-27 12:55:13,704 INFO [decode.py:316] Done!


**Congratulations!** You have successfully setup the environment and have run the first recipe in `icefall`_.

Have fun with ``icefall``!

YouTube Video
-------------

We provide the following YouTube video showing how to install `icefall`_.
It also shows how to debug various problems that you may encounter while
using `icefall`_.

.. note::

   To get the latest news of `next-gen Kaldi <https://github.com/k2-fsa>`_, please subscribe
   the following YouTube channel by `Nadira Povey <https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_:

      `<https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_

..  youtube:: LVmrBD0tLfE
