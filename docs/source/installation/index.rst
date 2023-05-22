.. _install icefall:

Installation
============



``icefall`` depends on `k2 <https://github.com/k2-fsa/k2>`_ and
`lhotse <https://github.com/lhotse-speech/lhotse>`_.

We recommend that you use the following steps to install the dependencies.

- (0) Install CUDA toolkit and cuDNN
- (1) Install PyTorch and torchaudio
- (2) Install k2
- (3) Install lhotse

.. caution::

   99% users who have issues about the installation are using conda.

.. caution::

   99% users who have issues about the installation are using conda.

.. caution::

   99% users who have issues about the installation are using conda.

.. hint::

   We suggest that you use ``pip install`` to install PyTorch.

   You can use the following command to create a virutal environment in Python:

    .. code-block:: bash

        python3 -m venv ./my_env
        source ./my_env/bin/activate

.. caution::

  Installation order matters.

(0) Install CUDA toolkit and cuDNN
----------------------------------

Please refer to
`<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
to install CUDA and cuDNN.


(1) Install PyTorch and torchaudio
----------------------------------

Please refer `<https://pytorch.org/>`_ to install PyTorch
and torchaudio.

.. hint::

   You can also go to  `<https://download.pytorch.org/whl/torch_stable.html>`_
   to download pre-compiled wheels and install them.

.. caution::

   Please install torch and torchaudio at the same time.


(2) Install k2
--------------

Please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install ``k2``.

.. caution::

  Please don't change your installed PyTorch after you have installed k2.

.. note::

   We suggest that you install k2 from source by following
   `<https://k2-fsa.github.io/k2/installation/from_source.html>`_
   or
   `<https://k2-fsa.github.io/k2/installation/for_developers.html>`_.

.. hint::

   Please always install the latest version of k2.

(3) Install lhotse
------------------

Please refer to `<https://lhotse.readthedocs.io/en/latest/getting-started.html#installation>`_
to install ``lhotse``.


.. hint::

    We strongly recommend you to use::

      pip install git+https://github.com/lhotse-speech/lhotse

    to install the latest version of lhotse.

(4) Download icefall
--------------------

``icefall`` is a collection of Python scripts; what you need is to download it
and set the environment variable ``PYTHONPATH`` to point to it.

Assume you want to place ``icefall`` in the folder ``/tmp``. The
following commands show you how to setup ``icefall``:


.. code-block:: bash

  cd /tmp
  git clone https://github.com/k2-fsa/icefall
  cd icefall
  pip install -r requirements.txt
  export PYTHONPATH=/tmp/icefall:$PYTHONPATH

.. HINT::

  You can put several versions of ``icefall`` in the same virtual environment.
  To switch among different versions of ``icefall``, just set ``PYTHONPATH``
  to point to the version you want.


Installation example
--------------------

The following shows an example about setting up the environment.


(1) Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ virtualenv -p python3.8  test-icefall

  created virtual environment CPython3.8.6.final.0-64 in 1540ms
    creator CPython3Posix(dest=/ceph-fj/fangjun/test-icefall, clear=False, no_vcs_ignore=False, global=False)
    seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/root/fangjun/.local/share/v
  irtualenv)
      added seed packages: pip==21.1.3, setuptools==57.4.0, wheel==0.36.2
    activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator


(2) Activate your virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ source test-icefall/bin/activate

(3) Install k2
~~~~~~~~~~~~~~

.. code-block:: bash

  $ pip install k2==1.4.dev20210822+cpu.torch1.9.0 -f https://k2-fsa.org/nightly/index.html

  Looking in links: https://k2-fsa.org/nightly/index.html
  Collecting k2==1.4.dev20210822+cpu.torch1.9.0
    Downloading https://k2-fsa.org/nightly/whl/k2-1.4.dev20210822%2Bcpu.torch1.9.0-cp38-cp38-linux_x86_64.whl (1.6 MB)
       |________________________________| 1.6 MB 185 kB/s
  Collecting graphviz
    Downloading graphviz-0.17-py3-none-any.whl (18 kB)
  Collecting torch==1.9.0
    Using cached torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl (831.4 MB)
  Collecting typing-extensions
    Using cached typing_extensions-3.10.0.0-py3-none-any.whl (26 kB)
  Installing collected packages: typing-extensions, torch, graphviz, k2
  Successfully installed graphviz-0.17 k2-1.4.dev20210822+cpu.torch1.9.0 torch-1.9.0 typing-extensions-3.10.0.0

.. WARNING::

  We choose to install a CPU version of k2 for testing. You would probably want to install
  a CUDA version of k2.


(4) Install lhotse
~~~~~~~~~~~~~~~~~~

.. code-block::

  $ pip install git+https://github.com/lhotse-speech/lhotse

  Collecting git+https://github.com/lhotse-speech/lhotse
    Cloning https://github.com/lhotse-speech/lhotse to /tmp/pip-req-build-7b1b76ge
    Running command git clone -q https://github.com/lhotse-speech/lhotse /tmp/pip-req-build-7b1b76ge
  Collecting audioread>=2.1.9
    Using cached audioread-2.1.9-py3-none-any.whl
  Collecting SoundFile>=0.10
    Using cached SoundFile-0.10.3.post1-py2.py3-none-any.whl (21 kB)
  Collecting click>=7.1.1
    Using cached click-8.0.1-py3-none-any.whl (97 kB)
  Collecting cytoolz>=0.10.1
    Using cached cytoolz-0.11.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.9 MB)
  Collecting dataclasses
    Using cached dataclasses-0.6-py3-none-any.whl (14 kB)
  Collecting h5py>=2.10.0
    Downloading h5py-3.4.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (4.5 MB)
       |________________________________| 4.5 MB 684 kB/s
  Collecting intervaltree>=3.1.0
    Using cached intervaltree-3.1.0-py2.py3-none-any.whl
  Collecting lilcom>=1.1.0
    Using cached lilcom-1.1.1-cp38-cp38-linux_x86_64.whl
  Collecting numpy>=1.18.1
    Using cached numpy-1.21.2-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.8 MB)
  Collecting packaging
    Using cached packaging-21.0-py3-none-any.whl (40 kB)
  Collecting pyyaml>=5.3.1
    Using cached PyYAML-5.4.1-cp38-cp38-manylinux1_x86_64.whl (662 kB)
  Collecting tqdm
    Downloading tqdm-4.62.1-py2.py3-none-any.whl (76 kB)
       |________________________________| 76 kB 2.7 MB/s
  Collecting torchaudio==0.9.0
    Downloading torchaudio-0.9.0-cp38-cp38-manylinux1_x86_64.whl (1.9 MB)
       |________________________________| 1.9 MB 73.1 MB/s
  Requirement already satisfied: torch==1.9.0 in ./test-icefall/lib/python3.8/site-packages (from torchaudio==0.9.0->lhotse===0.8.0.dev
  -2a1410b-clean) (1.9.0)
  Requirement already satisfied: typing-extensions in ./test-icefall/lib/python3.8/site-packages (from torch==1.9.0->torchaudio==0.9.0-
  >lhotse===0.8.0.dev-2a1410b-clean) (3.10.0.0)
  Collecting toolz>=0.8.0
    Using cached toolz-0.11.1-py3-none-any.whl (55 kB)
  Collecting sortedcontainers<3.0,>=2.0
    Using cached sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
  Collecting cffi>=1.0
    Using cached cffi-1.14.6-cp38-cp38-manylinux1_x86_64.whl (411 kB)
  Collecting pycparser
    Using cached pycparser-2.20-py2.py3-none-any.whl (112 kB)
  Collecting pyparsing>=2.0.2
    Using cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
  Building wheels for collected packages: lhotse
    Building wheel for lhotse (setup.py) ... done
    Created wheel for lhotse: filename=lhotse-0.8.0.dev_2a1410b_clean-py3-none-any.whl size=342242 sha256=f683444afa4dc0881133206b4646a
  9d0f774224cc84000f55d0a67f6e4a37997
    Stored in directory: /tmp/pip-ephem-wheel-cache-ftu0qysz/wheels/7f/7a/8e/a0bf241336e2e3cb573e1e21e5600952d49f5162454f2e612f
    WARNING: Built wheel for lhotse is invalid: Metadata 1.2 mandates PEP 440 version, but '0.8.0.dev-2a1410b-clean' is not
  Failed to build lhotse
  Installing collected packages: pycparser, toolz, sortedcontainers, pyparsing, numpy, cffi, tqdm, torchaudio, SoundFile, pyyaml, packa
  ging, lilcom, intervaltree, h5py, dataclasses, cytoolz, click, audioread, lhotse
      Running setup.py install for lhotse ... done
    DEPRECATION: lhotse was installed using the legacy 'setup.py install' method, because a wheel could not be built for it. A possible
   replacement is to fix the wheel build issue reported above. You can find discussion regarding this at https://github.com/pypa/pip/is
  sues/8368.
  Successfully installed SoundFile-0.10.3.post1 audioread-2.1.9 cffi-1.14.6 click-8.0.1 cytoolz-0.11.0 dataclasses-0.6 h5py-3.4.0 inter
  valtree-3.1.0 lhotse-0.8.0.dev-2a1410b-clean lilcom-1.1.1 numpy-1.21.2 packaging-21.0 pycparser-2.20 pyparsing-2.4.7 pyyaml-5.4.1 sor
  tedcontainers-2.4.0 toolz-0.11.1 torchaudio-0.9.0 tqdm-4.62.1

(5) Download icefall
~~~~~~~~~~~~~~~~~~~~

.. code-block::

  $ cd /tmp
  $ git clone https://github.com/k2-fsa/icefall

  Cloning into 'icefall'...
  remote: Enumerating objects: 500, done.
  remote: Counting objects: 100% (500/500), done.
  remote: Compressing objects: 100% (308/308), done.
  remote: Total 500 (delta 263), reused 307 (delta 102), pack-reused 0
  Receiving objects: 100% (500/500), 172.49 KiB | 385.00 KiB/s, done.
  Resolving deltas: 100% (263/263), done.

  $ cd icefall
  $ pip install -r requirements.txt

  Collecting kaldilm
    Downloading kaldilm-1.8.tar.gz (48 kB)
       |________________________________| 48 kB 574 kB/s
  Collecting kaldialign
    Using cached kaldialign-0.2-cp38-cp38-linux_x86_64.whl
  Collecting sentencepiece>=0.1.96
    Using cached sentencepiece-0.1.96-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
  Collecting tensorboard
    Using cached tensorboard-2.6.0-py3-none-any.whl (5.6 MB)
  Requirement already satisfied: setuptools>=41.0.0 in /ceph-fj/fangjun/test-icefall/lib/python3.8/site-packages (from tensorboard->-r
  requirements.txt (line 4)) (57.4.0)
  Collecting absl-py>=0.4
    Using cached absl_py-0.13.0-py3-none-any.whl (132 kB)
  Collecting google-auth-oauthlib<0.5,>=0.4.1
    Using cached google_auth_oauthlib-0.4.5-py2.py3-none-any.whl (18 kB)
  Collecting grpcio>=1.24.3
    Using cached grpcio-1.39.0-cp38-cp38-manylinux2014_x86_64.whl (4.3 MB)
  Requirement already satisfied: wheel>=0.26 in /ceph-fj/fangjun/test-icefall/lib/python3.8/site-packages (from tensorboard->-r require
  ments.txt (line 4)) (0.36.2)
  Requirement already satisfied: numpy>=1.12.0 in /ceph-fj/fangjun/test-icefall/lib/python3.8/site-packages (from tensorboard->-r requi
  rements.txt (line 4)) (1.21.2)
  Collecting protobuf>=3.6.0
    Using cached protobuf-3.17.3-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
  Collecting werkzeug>=0.11.15
    Using cached Werkzeug-2.0.1-py3-none-any.whl (288 kB)
  Collecting tensorboard-data-server<0.7.0,>=0.6.0
    Using cached tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
  Collecting google-auth<2,>=1.6.3
    Downloading google_auth-1.35.0-py2.py3-none-any.whl (152 kB)
       |________________________________| 152 kB 1.4 MB/s
  Collecting requests<3,>=2.21.0
    Using cached requests-2.26.0-py2.py3-none-any.whl (62 kB)
  Collecting tensorboard-plugin-wit>=1.6.0
    Using cached tensorboard_plugin_wit-1.8.0-py3-none-any.whl (781 kB)
  Collecting markdown>=2.6.8
    Using cached Markdown-3.3.4-py3-none-any.whl (97 kB)
  Collecting six
    Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
  Collecting cachetools<5.0,>=2.0.0
    Using cached cachetools-4.2.2-py3-none-any.whl (11 kB)
  Collecting rsa<5,>=3.1.4
    Using cached rsa-4.7.2-py3-none-any.whl (34 kB)
  Collecting pyasn1-modules>=0.2.1
    Using cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
  Collecting requests-oauthlib>=0.7.0
    Using cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)
  Collecting pyasn1<0.5.0,>=0.4.6
    Using cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
  Collecting urllib3<1.27,>=1.21.1
    Using cached urllib3-1.26.6-py2.py3-none-any.whl (138 kB)
  Collecting certifi>=2017.4.17
    Using cached certifi-2021.5.30-py2.py3-none-any.whl (145 kB)
  Collecting charset-normalizer~=2.0.0
    Using cached charset_normalizer-2.0.4-py3-none-any.whl (36 kB)
  Collecting idna<4,>=2.5
    Using cached idna-3.2-py3-none-any.whl (59 kB)
  Collecting oauthlib>=3.0.0
    Using cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)
  Building wheels for collected packages: kaldilm
    Building wheel for kaldilm (setup.py) ... done
    Created wheel for kaldilm: filename=kaldilm-1.8-cp38-cp38-linux_x86_64.whl size=897233 sha256=eccb906cafcd45bf9a7e1a1718e4534254bfb
  f4c0d0cbc66eee6c88d68a63862
    Stored in directory: /root/fangjun/.cache/pip/wheels/85/7d/63/f2dd586369b8797cb36d213bf3a84a789eeb92db93d2e723c9
  Successfully built kaldilm
  Installing collected packages: urllib3, pyasn1, idna, charset-normalizer, certifi, six, rsa, requests, pyasn1-modules, oauthlib, cach
  etools, requests-oauthlib, google-auth, werkzeug, tensorboard-plugin-wit, tensorboard-data-server, protobuf, markdown, grpcio, google
  -auth-oauthlib, absl-py, tensorboard, sentencepiece, kaldilm, kaldialign
  Successfully installed absl-py-0.13.0 cachetools-4.2.2 certifi-2021.5.30 charset-normalizer-2.0.4 google-auth-1.35.0 google-auth-oaut
  hlib-0.4.5 grpcio-1.39.0 idna-3.2 kaldialign-0.2 kaldilm-1.8 markdown-3.3.4 oauthlib-3.1.1 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-module
  s-0.2.8 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 sentencepiece-0.1.96 six-1.16.0 tensorboard-2.6.0 tensorboard-data-server-0
  .6.1 tensorboard-plugin-wit-1.8.0 urllib3-1.26.6 werkzeug-2.0.1


Test Your Installation
----------------------

To test that your installation is successful, let us run
the `yesno recipe <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR>`_
on CPU.

Data preparation
~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ export PYTHONPATH=/tmp/icefall:$PYTHONPATH
  $ cd /tmp/icefall
  $ cd egs/yesno/ASR
  $ ./prepare.sh

The log of running ``./prepare.sh`` is:

.. code-block::

   2023-05-12 17:55:21 (prepare.sh:27:main) dl_dir: /tmp/icefall/egs/yesno/ASR/download
   2023-05-12 17:55:21 (prepare.sh:30:main) Stage 0: Download data
   /tmp/icefall/egs/yesno/ASR/download/waves_yesno.tar.gz: 100%|_______________________________________________________________| 4.70M/4.70M [06:54<00:00, 11.4kB/s]
   2023-05-12 18:02:19 (prepare.sh:39:main) Stage 1: Prepare yesno manifest
   2023-05-12 18:02:21 (prepare.sh:45:main) Stage 2: Compute fbank for yesno
   2023-05-12 18:02:23,199 INFO [compute_fbank_yesno.py:65] Processing train
   Extracting and storing features: 100%|_______________________________________________________________| 90/90 [00:00<00:00, 212.60it/s]
   2023-05-12 18:02:23,640 INFO [compute_fbank_yesno.py:65] Processing test
   Extracting and storing features: 100%|_______________________________________________________________| 30/30 [00:00<00:00, 304.53it/s]
   2023-05-12 18:02:24 (prepare.sh:51:main) Stage 3: Prepare lang
   2023-05-12 18:02:26 (prepare.sh:66:main) Stage 4: Prepare G
   /project/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Read(std::istream&):79
   [I] Reading \data\ section.
   /project/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Read(std::istream&):140
   [I] Reading \1-grams: section.
   2023-05-12 18:02:26 (prepare.sh:92:main) Stage 5: Compile HLG
   2023-05-12 18:02:28,581 INFO [compile_hlg.py:124] Processing data/lang_phone
   2023-05-12 18:02:28,582 INFO [lexicon.py:171] Converting L.pt to Linv.pt
   2023-05-12 18:02:28,609 INFO [compile_hlg.py:48] Building ctc_topo. max_token_id: 3
   2023-05-12 18:02:28,610 INFO [compile_hlg.py:52] Loading G.fst.txt
   2023-05-12 18:02:28,611 INFO [compile_hlg.py:62] Intersecting L and G
   2023-05-12 18:02:28,613 INFO [compile_hlg.py:64] LG shape: (4, None)
   2023-05-12 18:02:28,613 INFO [compile_hlg.py:66] Connecting LG
   2023-05-12 18:02:28,614 INFO [compile_hlg.py:68] LG shape after k2.connect: (4, None)
   2023-05-12 18:02:28,614 INFO [compile_hlg.py:70] <class 'torch.Tensor'>
   2023-05-12 18:02:28,614 INFO [compile_hlg.py:71] Determinizing LG
   2023-05-12 18:02:28,615 INFO [compile_hlg.py:74] <class '_k2.ragged.RaggedTensor'>
   2023-05-12 18:02:28,615 INFO [compile_hlg.py:76] Connecting LG after k2.determinize
   2023-05-12 18:02:28,615 INFO [compile_hlg.py:79] Removing disambiguation symbols on LG
   2023-05-12 18:02:28,616 INFO [compile_hlg.py:91] LG shape after k2.remove_epsilon: (6, None)
   2023-05-12 18:02:28,617 INFO [compile_hlg.py:96] Arc sorting LG
   2023-05-12 18:02:28,617 INFO [compile_hlg.py:99] Composing H and LG
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:106] Connecting LG
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:109] Arc sorting LG
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:111] HLG.shape: (8, None)
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:127] Saving HLG.pt to data/lang_phone


Training
~~~~~~~~

Now let us run the training part:

.. code-block::

  $ export CUDA_VISIBLE_DEVICES=""
  $ ./tdnn/train.py

.. CAUTION::

  We use ``export CUDA_VISIBLE_DEVICES=""`` so that ``icefall`` uses CPU
  even if there are GPUs available.

.. hint::

   In case you get a ``Segmentation fault (core dump)`` error, please use:

      .. code-block:: bash

        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

   See more at `<https://github.com/k2-fsa/icefall/issues/674>` if you are
   interested.

The training log is given below:

.. code-block::

   2023-05-12 18:04:59,759 INFO [train.py:481] Training started
   2023-05-12 18:04:59,759 INFO [train.py:482] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01, 'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 
   'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, 'best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'reset_interval': 20, 'valid_interval': 10, 'beam_size': 10, 
   'reduction': 'sum', 'use_double_scores': True, 'world_size': 1, 'master_port': 12354, 'tensorboard': True, 'num_epochs': 15, 'seed': 42, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 30.0,
   'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 'num_workers': 2, 
   'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '3b7f09fa35e72589914f67089c0da9f196a92ca4', 'k2-git-date': 'Mon May 8 22:58:45 2023', 
   'lhotse-version': '1.15.0.dev+git.6fcfced.clean', 'torch-version': '2.0.0+cu118', 'torch-cuda-available': False, 'torch-cuda-version': '11.8', 'python-version': '3.1', 'icefall-git-branch': 'master', 
   'icefall-git-sha1': '30bde4b-clean', 'icefall-git-date': 'Thu May 11 17:37:47 2023', 'icefall-path': '/tmp/icefall', 
   'k2-path': 'tmp/lib/python3.10/site-packages/k2-1.24.3.dev20230512+cuda11.8.torch2.0.0-py3.10-linux-x86_64.egg/k2/__init__.py', 
   'lhotse-path': 'tmp/lib/python3.10/site-packages/lhotse/__init__.py', 'hostname': 'host', 'IP address': '0.0.0.0'}}
   2023-05-12 18:04:59,761 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
   2023-05-12 18:04:59,764 INFO [train.py:495] device: cpu
   2023-05-12 18:04:59,791 INFO [asr_datamodule.py:146] About to get train cuts
   2023-05-12 18:04:59,791 INFO [asr_datamodule.py:244] About to get train cuts
   2023-05-12 18:04:59,852 INFO [asr_datamodule.py:149] About to create train dataset
   2023-05-12 18:04:59,852 INFO [asr_datamodule.py:199] Using SingleCutSampler.
   2023-05-12 18:04:59,852 INFO [asr_datamodule.py:205] About to create train dataloader
   2023-05-12 18:04:59,853 INFO [asr_datamodule.py:218] About to get test cuts
   2023-05-12 18:04:59,853 INFO [asr_datamodule.py:252] About to get test cuts
   2023-05-12 18:04:59,986 INFO [train.py:422] Epoch 0, batch 0, loss[loss=1.065, over 2436.00 frames. ], tot_loss[loss=1.065, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:00,352 INFO [train.py:422] Epoch 0, batch 10, loss[loss=0.4561, over 2828.00 frames. ], tot_loss[loss=0.7076, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:00,691 INFO [train.py:444] Epoch 0, validation loss=0.9002, over 18067.00 frames.
   2023-05-12 18:05:00,996 INFO [train.py:422] Epoch 0, batch 20, loss[loss=0.2555, over 2695.00 frames. ], tot_loss[loss=0.484, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:01,217 INFO [train.py:444] Epoch 0, validation loss=0.4688, over 18067.00 frames.
   2023-05-12 18:05:01,251 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-0.pt
   2023-05-12 18:05:01,389 INFO [train.py:422] Epoch 1, batch 0, loss[loss=0.2532, over 2436.00 frames. ], tot_loss[loss=0.2532, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:01,637 INFO [train.py:422] Epoch 1, batch 10, loss[loss=0.1139, over 2828.00 frames. ], tot_loss[loss=0.1592, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:01,859 INFO [train.py:444] Epoch 1, validation loss=0.1629, over 18067.00 frames.
   2023-05-12 18:05:02,094 INFO [train.py:422] Epoch 1, batch 20, loss[loss=0.0767, over 2695.00 frames. ], tot_loss[loss=0.118, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:02,350 INFO [train.py:444] Epoch 1, validation loss=0.06778, over 18067.00 frames.
   2023-05-12 18:05:02,395 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-1.pt

  ... ...

   2023-05-12 18:05:14,789 INFO [train.py:422] Epoch 13, batch 0, loss[loss=0.01056, over 2436.00 frames. ], tot_loss[loss=0.01056, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:15,016 INFO [train.py:422] Epoch 13, batch 10, loss[loss=0.009022, over 2828.00 frames. ], tot_loss[loss=0.009985, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:15,271 INFO [train.py:444] Epoch 13, validation loss=0.01088, over 18067.00 frames.
   2023-05-12 18:05:15,497 INFO [train.py:422] Epoch 13, batch 20, loss[loss=0.01174, over 2695.00 frames. ], tot_loss[loss=0.01077, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:15,747 INFO [train.py:444] Epoch 13, validation loss=0.01087, over 18067.00 frames.
   2023-05-12 18:05:15,783 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-13.pt
   2023-05-12 18:05:15,921 INFO [train.py:422] Epoch 14, batch 0, loss[loss=0.01045, over 2436.00 frames. ], tot_loss[loss=0.01045, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:16,146 INFO [train.py:422] Epoch 14, batch 10, loss[loss=0.008957, over 2828.00 frames. ], tot_loss[loss=0.009903, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:16,374 INFO [train.py:444] Epoch 14, validation loss=0.01092, over 18067.00 frames.
   2023-05-12 18:05:16,598 INFO [train.py:422] Epoch 14, batch 20, loss[loss=0.01169, over 2695.00 frames. ], tot_loss[loss=0.01065, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:16,824 INFO [train.py:444] Epoch 14, validation loss=0.01077, over 18067.00 frames.
   2023-05-12 18:05:16,862 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-14.pt
   2023-05-12 18:05:16,865 INFO [train.py:555] Done!

Decoding
~~~~~~~~

Let us use the trained model to decode the test set:

.. code-block::

  $ ./tdnn/decode.py

The decoding log is:

.. code-block::

   2023-05-12 18:08:30,482 INFO [decode.py:263] Decoding started
   2023-05-12 18:08:30,483 INFO [decode.py:264] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lm_dir': PosixPath('data/lm'), 'feature_dim': 23, 
   'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 14, 'avg': 2, 'export': False, 'feature_dir': PosixPath('data/fbank'), 
   'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 
   'num_workers': 2, 'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '3b7f09fa35e72589914f67089c0da9f196a92ca4', 'k2-git-date': 'Mon May 8 22:58:45 2023', 
   'lhotse-version': '1.15.0.dev+git.6fcfced.clean', 'torch-version': '2.0.0+cu118', 'torch-cuda-available': False, 'torch-cuda-version': '11.8', 'python-version': '3.1', 'icefall-git-branch': 'master', 
   'icefall-git-sha1': '30bde4b-clean', 'icefall-git-date': 'Thu May 11 17:37:47 2023', 'icefall-path': '/tmp/icefall', 
   'k2-path': '/tmp/lib/python3.10/site-packages/k2-1.24.3.dev20230512+cuda11.8.torch2.0.0-py3.10-linux-x86_64.egg/k2/__init__.py', 
   'lhotse-path': '/tmp/lib/python3.10/site-packages/lhotse/__init__.py', 'hostname': 'host', 'IP address': '0.0.0.0'}}
   2023-05-12 18:08:30,483 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
   2023-05-12 18:08:30,487 INFO [decode.py:273] device: cpu
   2023-05-12 18:08:30,513 INFO [decode.py:291] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
   2023-05-12 18:08:30,521 INFO [asr_datamodule.py:218] About to get test cuts
   2023-05-12 18:08:30,521 INFO [asr_datamodule.py:252] About to get test cuts
   2023-05-12 18:08:30,675 INFO [decode.py:204] batch 0/?, cuts processed until now is 4
   2023-05-12 18:08:30,923 INFO [decode.py:241] The transcripts are stored in tdnn/exp/recogs-test_set.txt
   2023-05-12 18:08:30,924 INFO [utils.py:558] [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
   2023-05-12 18:08:30,925 INFO [decode.py:249] Wrote detailed error stats to tdnn/exp/errs-test_set.txt
   2023-05-12 18:08:30,925 INFO [decode.py:316] Done!

**Congratulations!** You have successfully setup the environment and have run the first recipe in ``icefall``.

Have fun with ``icefall``!

YouTube Video
-------------

We provide the following YouTube video showing how to install ``icefall``.
It also shows how to debug various problems that you may encounter while
using ``icefall``.

.. note::

   To get the latest news of `next-gen Kaldi <https://github.com/k2-fsa>`_, please subscribe
   the following YouTube channel by `Nadira Povey <https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_:

      `<https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_

..  youtube:: LVmrBD0tLfE
