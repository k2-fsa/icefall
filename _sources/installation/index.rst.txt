.. _install icefall:

Installation
============

- |os|
- |device|
- |python_versions|
- |torch_versions|
- |k2_versions|

.. |os| image:: ./images/os-Linux_macOS-ff69b4.svg
  :alt: Supported operating systems

.. |device| image:: ./images/device-CPU_CUDA-orange.svg
  :alt: Supported devices

.. |python_versions| image:: ./images/python-gt-v3.6-blue.svg
  :alt: Supported python versions

.. |torch_versions| image:: ./images/torch-gt-v1.6.0-green.svg
  :alt: Supported PyTorch versions

.. |k2_versions| image:: ./images/k2-gt-v1.9-blueviolet.svg
  :alt: Supported k2 versions

``icefall`` depends on `k2 <https://github.com/k2-fsa/k2>`_ and
`lhotse <https://github.com/lhotse-speech/lhotse>`_.

We recommend you to use the following steps to install the dependencies.

- (0) Install PyTorch and torchaudio
- (1) Install k2
- (2) Install lhotse

.. caution::

  Installation order matters.

(0) Install PyTorch and torchaudio
----------------------------------

Please refer `<https://pytorch.org/>`_ to install PyTorch
and torchaudio.


(1) Install k2
--------------

Please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install ``k2``.

.. CAUTION::

  You need to install ``k2`` with a version at least **v1.9**.

.. HINT::

  If you have already installed PyTorch and don't want to replace it,
  please install a version of ``k2`` that is compiled against the version
  of PyTorch you are using.

(2) Install lhotse
------------------

Please refer to `<https://lhotse.readthedocs.io/en/latest/getting-started.html#installation>`_
to install ``lhotse``.


.. hint::

    We strongly recommend you to use::

      pip install git+https://github.com/lhotse-speech/lhotse

    to install the latest version of lhotse.


(3) Download icefall
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

  2021-08-23 19:27:26 (prepare.sh:24:main) dl_dir: /tmp/icefall/egs/yesno/ASR/download
  2021-08-23 19:27:26 (prepare.sh:27:main) stage 0: Download data
  Downloading waves_yesno.tar.gz: 4.49MB [00:03, 1.39MB/s]
  2021-08-23 19:27:30 (prepare.sh:36:main) Stage 1: Prepare yesno manifest
  2021-08-23 19:27:31 (prepare.sh:42:main) Stage 2: Compute fbank for yesno
  2021-08-23 19:27:32,803 INFO [compute_fbank_yesno.py:52] Processing train
  Extracting and storing features: 100%|_______________________________________________________________| 90/90 [00:01<00:00, 80.57it/s]
  2021-08-23 19:27:34,085 INFO [compute_fbank_yesno.py:52] Processing test
  Extracting and storing features: 100%|______________________________________________________________| 30/30 [00:00<00:00, 248.21it/s]
  2021-08-23 19:27:34 (prepare.sh:48:main) Stage 3: Prepare lang
  2021-08-23 19:27:35 (prepare.sh:63:main) Stage 4: Prepare G
  /tmp/pip-install-fcordre9/kaldilm_6899d26f2d684ad48f21025950cd2866/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Rea
  d(std::istream&):79
  [I] Reading \data\ section.
  /tmp/pip-install-fcordre9/kaldilm_6899d26f2d684ad48f21025950cd2866/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Rea
  d(std::istream&):140
  [I] Reading \1-grams: section.
  2021-08-23 19:27:35 (prepare.sh:89:main) Stage 5: Compile HLG
  2021-08-23 19:27:35,928 INFO [compile_hlg.py:120] Processing data/lang_phone
  2021-08-23 19:27:35,929 INFO [lexicon.py:116] Converting L.pt to Linv.pt
  2021-08-23 19:27:35,931 INFO [compile_hlg.py:48] Building ctc_topo. max_token_id: 3
  2021-08-23 19:27:35,932 INFO [compile_hlg.py:52] Loading G.fst.txt
  2021-08-23 19:27:35,932 INFO [compile_hlg.py:62] Intersecting L and G
  2021-08-23 19:27:35,933 INFO [compile_hlg.py:64] LG shape: (4, None)
  2021-08-23 19:27:35,933 INFO [compile_hlg.py:66] Connecting LG
  2021-08-23 19:27:35,933 INFO [compile_hlg.py:68] LG shape after k2.connect: (4, None)
  2021-08-23 19:27:35,933 INFO [compile_hlg.py:70] <class 'torch.Tensor'>
  2021-08-23 19:27:35,933 INFO [compile_hlg.py:71] Determinizing LG
  2021-08-23 19:27:35,934 INFO [compile_hlg.py:74] <class '_k2.RaggedInt'>
  2021-08-23 19:27:35,934 INFO [compile_hlg.py:76] Connecting LG after k2.determinize
  2021-08-23 19:27:35,934 INFO [compile_hlg.py:79] Removing disambiguation symbols on LG
  2021-08-23 19:27:35,934 INFO [compile_hlg.py:87] LG shape after k2.remove_epsilon: (6, None)
  2021-08-23 19:27:35,935 INFO [compile_hlg.py:92] Arc sorting LG
  2021-08-23 19:27:35,935 INFO [compile_hlg.py:95] Composing H and LG
  2021-08-23 19:27:35,935 INFO [compile_hlg.py:102] Connecting LG
  2021-08-23 19:27:35,935 INFO [compile_hlg.py:105] Arc sorting LG
  2021-08-23 19:27:35,936 INFO [compile_hlg.py:107] HLG.shape: (8, None)
  2021-08-23 19:27:35,936 INFO [compile_hlg.py:123] Saving HLG.pt to data/lang_phone


Training
~~~~~~~~

Now let us run the training part:

.. code-block::

  $ export CUDA_VISIBLE_DEVICES=""
  $ ./tdnn/train.py

.. CAUTION::

  We use ``export CUDA_VISIBLE_DEVICES=""`` so that ``icefall`` uses CPU
  even if there are GPUs available.

The training log is given below:

.. code-block::

  2021-08-23 19:30:31,072 INFO [train.py:465] Training started
  2021-08-23 19:30:31,072 INFO [train.py:466] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01,
  'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, '
  best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'valid_interval': 10, 'beam_size': 10, 'reduction': 'sum', 'use_doub
  le_scores': True, 'world_size': 1, 'master_port': 12354, 'tensorboard': True, 'num_epochs': 15, 'feature_dir': PosixPath('data/fbank'
  ), 'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0
  , 'on_the_fly_feats': False, 'shuffle': True, 'return_cuts': True, 'num_workers': 2}
  2021-08-23 19:30:31,074 INFO [lexicon.py:113] Loading pre-compiled data/lang_phone/Linv.pt
  2021-08-23 19:30:31,098 INFO [asr_datamodule.py:146] About to get train cuts
  2021-08-23 19:30:31,098 INFO [asr_datamodule.py:240] About to get train cuts
  2021-08-23 19:30:31,102 INFO [asr_datamodule.py:149] About to create train dataset
  2021-08-23 19:30:31,102 INFO [asr_datamodule.py:200] Using SingleCutSampler.
  2021-08-23 19:30:31,102 INFO [asr_datamodule.py:206] About to create train dataloader
  2021-08-23 19:30:31,102 INFO [asr_datamodule.py:219] About to get test cuts
  2021-08-23 19:30:31,102 INFO [asr_datamodule.py:246] About to get test cuts
  2021-08-23 19:30:31,357 INFO [train.py:416] Epoch 0, batch 0, batch avg loss 1.0789, total avg loss: 1.0789, batch size: 4
  2021-08-23 19:30:31,848 INFO [train.py:416] Epoch 0, batch 10, batch avg loss 0.5356, total avg loss: 0.7556, batch size: 4
  2021-08-23 19:30:32,301 INFO [train.py:432] Epoch 0, valid loss 0.9972, best valid loss: 0.9972 best valid epoch: 0
  2021-08-23 19:30:32,805 INFO [train.py:416] Epoch 0, batch 20, batch avg loss 0.2436, total avg loss: 0.5717, batch size: 3
  2021-08-23 19:30:33,109 INFO [train.py:432] Epoch 0, valid loss 0.4167, best valid loss: 0.4167 best valid epoch: 0
  2021-08-23 19:30:33,121 INFO [checkpoint.py:62] Saving checkpoint to tdnn/exp/epoch-0.pt
  2021-08-23 19:30:33,325 INFO [train.py:416] Epoch 1, batch 0, batch avg loss 0.2214, total avg loss: 0.2214, batch size: 5
  2021-08-23 19:30:33,798 INFO [train.py:416] Epoch 1, batch 10, batch avg loss 0.0781, total avg loss: 0.1343, batch size: 5
  2021-08-23 19:30:34,065 INFO [train.py:432] Epoch 1, valid loss 0.0859, best valid loss: 0.0859 best valid epoch: 1
  2021-08-23 19:30:34,556 INFO [train.py:416] Epoch 1, batch 20, batch avg loss 0.0421, total avg loss: 0.0975, batch size: 3
  2021-08-23 19:30:34,810 INFO [train.py:432] Epoch 1, valid loss 0.0431, best valid loss: 0.0431 best valid epoch: 1
  2021-08-23 19:30:34,824 INFO [checkpoint.py:62] Saving checkpoint to tdnn/exp/epoch-1.pt

  ... ...

  2021-08-23 19:30:49,657 INFO [train.py:416] Epoch 13, batch 0, batch avg loss 0.0109, total avg loss: 0.0109, batch size: 5
  2021-08-23 19:30:49,984 INFO [train.py:416] Epoch 13, batch 10, batch avg loss 0.0093, total avg loss: 0.0096, batch size: 4
  2021-08-23 19:30:50,239 INFO [train.py:432] Epoch 13, valid loss 0.0104, best valid loss: 0.0101 best valid epoch: 12
  2021-08-23 19:30:50,569 INFO [train.py:416] Epoch 13, batch 20, batch avg loss 0.0092, total avg loss: 0.0096, batch size: 2
  2021-08-23 19:30:50,819 INFO [train.py:432] Epoch 13, valid loss 0.0101, best valid loss: 0.0101 best valid epoch: 13
  2021-08-23 19:30:50,835 INFO [checkpoint.py:62] Saving checkpoint to tdnn/exp/epoch-13.pt
  2021-08-23 19:30:51,024 INFO [train.py:416] Epoch 14, batch 0, batch avg loss 0.0105, total avg loss: 0.0105, batch size: 5
  2021-08-23 19:30:51,317 INFO [train.py:416] Epoch 14, batch 10, batch avg loss 0.0099, total avg loss: 0.0097, batch size: 4
  2021-08-23 19:30:51,552 INFO [train.py:432] Epoch 14, valid loss 0.0108, best valid loss: 0.0101 best valid epoch: 13
  2021-08-23 19:30:51,869 INFO [train.py:416] Epoch 14, batch 20, batch avg loss 0.0096, total avg loss: 0.0097, batch size: 5
  2021-08-23 19:30:52,107 INFO [train.py:432] Epoch 14, valid loss 0.0102, best valid loss: 0.0101 best valid epoch: 13
  2021-08-23 19:30:52,126 INFO [checkpoint.py:62] Saving checkpoint to tdnn/exp/epoch-14.pt
  2021-08-23 19:30:52,128 INFO [train.py:537] Done!

Decoding
~~~~~~~~

Let us use the trained model to decode the test set:

.. code-block::

  $ ./tdnn/decode.py

The decoding log is:

.. code-block::

  2021-08-23 19:35:30,192 INFO [decode.py:249] Decoding started
  2021-08-23 19:35:30,192 INFO [decode.py:250] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lm_dir': PosixPath('data/lm'), 'feature_dim': 23, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 14, 'avg': 2, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': True, 'return_cuts': True, 'num_workers': 2}
  2021-08-23 19:35:30,193 INFO [lexicon.py:113] Loading pre-compiled data/lang_phone/Linv.pt
  2021-08-23 19:35:30,213 INFO [decode.py:259] device: cpu
  2021-08-23 19:35:30,217 INFO [decode.py:279] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
  /tmp/icefall/icefall/checkpoint.py:146: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch.
  It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
  To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /pytorch/aten/src/ATen/native/BinaryOps.cpp:450.)
    avg[k] //= n
  2021-08-23 19:35:30,220 INFO [asr_datamodule.py:219] About to get test cuts
  2021-08-23 19:35:30,220 INFO [asr_datamodule.py:246] About to get test cuts
  2021-08-23 19:35:30,409 INFO [decode.py:190] batch 0/8, cuts processed until now is 4
  2021-08-23 19:35:30,571 INFO [decode.py:228] The transcripts are stored in tdnn/exp/recogs-test_set.txt
  2021-08-23 19:35:30,572 INFO [utils.py:317] [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
  2021-08-23 19:35:30,573 INFO [decode.py:236] Wrote detailed error stats to tdnn/exp/errs-test_set.txt
  2021-08-23 19:35:30,573 INFO [decode.py:299] Done!

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
