Model Export
============

There are three ways to export a pre-trained model.

  - Export the model parameters via `model.state_dict() <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.state_dict>`_
  - Export via `torchscript <https://pytorch.org/docs/stable/jit.html>`_: either `torch.jit.script() <https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script>`_ or `torch.jit.trace() <https://pytorch.org/docs/stable/generated/torch.jit.trace.html>`_
  - Export to `ONNX`_ via `torch.onnx.export() <https://pytorch.org/docs/stable/onnx.html>`_

Each method is explained below in detail.

Export the model parameters via model.state_dict()
---------------------------------------------------

The command for this kind of export is

.. code-block:: bash

   cd /tmp/icefall
   export PYTHONPATH=/tmp/icefall:$PYTHONPATH
   cd egs/yesno/ASR

   # assume that "--epoch 14 --avg 2" produces the lowest WER.

   ./tdnn/export.py --epoch 14 --avg 2

The output logs are given below:

.. code-block:: bash

  2023-08-16 20:42:03,912 INFO [export.py:76] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01, 'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, 'best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'reset_interval': 20, 'valid_interval': 10, 'beam_size': 10, 'reduction': 'sum', 'use_double_scores': True, 'epoch': 14, 'avg': 2, 'jit': False}
  2023-08-16 20:42:03,913 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
  2023-08-16 20:42:03,950 INFO [export.py:93] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
  2023-08-16 20:42:03,971 INFO [export.py:106] Not using torch.jit.script
  2023-08-16 20:42:03,974 INFO [export.py:111] Saved to tdnn/exp/pretrained.pt

We can see from the logs that the exported model is saved to the file ``tdnn/exp/pretrained.pt``.

To give you an idea of what ``tdnn/exp/pretrained.pt`` contains, we can use the following command:

.. code-block:: python3

    >>> import torch
    >>> m = torch.load("tdnn/exp/pretrained.pt", weights_only=False)
    >>> list(m.keys())
    ['model']
    >>> list(m["model"].keys())
    ['tdnn.0.weight', 'tdnn.0.bias', 'tdnn.2.running_mean', 'tdnn.2.running_var', 'tdnn.2.num_batches_tracked', 'tdnn.3.weight', 'tdnn.3.bias', 'tdnn.5.running_mean', 'tdnn.5.running_var', 'tdnn.5.num_batches_tracked', 'tdnn.6.weight', 'tdnn.6.bias', 'tdnn.8.running_mean', 'tdnn.8.running_var', 'tdnn.8.num_batches_tracked', 'output_linear.weight', 'output_linear.bias']

We can use ``tdnn/exp/pretrained.pt`` in the following way with ``./tdnn/decode.py``:

.. code-block:: bash

   cd tdnn/exp
   ln -s pretrained.pt epoch-99.pt
   cd ../..

   ./tdnn/decode.py --epoch 99 --avg 1

The output logs of the above command are given below:

.. code-block:: bash

    2023-08-16 20:45:48,089 INFO [decode.py:262] Decoding started
    2023-08-16 20:45:48,090 INFO [decode.py:263] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'feature_dim': 23, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 99, 'avg': 1, 'export': False, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 'num_workers': 2, 'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': False, 'k2-git-sha1': 'ad79f1c699c684de9785ed6ca5edb805a41f78c3', 'k2-git-date': 'Wed Jul 26 09:30:42 2023', 'lhotse-version': '1.16.0.dev+git.aa073f6.clean', 'torch-version': '2.0.0', 'torch-cuda-available': False, 'torch-cuda-version': None, 'python-version': '3.1', 'icefall-git-branch': 'master', 'icefall-git-sha1': '9a47c08-clean', 'icefall-git-date': 'Mon Aug 14 22:10:50 2023', 'icefall-path': '/private/tmp/icefall', 'k2-path': '/private/tmp/icefall_env/lib/python3.11/site-packages/k2/__init__.py', 'lhotse-path': '/private/tmp/icefall_env/lib/python3.11/site-packages/lhotse/__init__.py', 'hostname': 'fangjuns-MacBook-Pro.local', 'IP address': '127.0.0.1'}}
    2023-08-16 20:45:48,092 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
    2023-08-16 20:45:48,103 INFO [decode.py:272] device: cpu
    2023-08-16 20:45:48,109 INFO [checkpoint.py:112] Loading checkpoint from tdnn/exp/epoch-99.pt
    2023-08-16 20:45:48,115 INFO [asr_datamodule.py:218] About to get test cuts
    2023-08-16 20:45:48,115 INFO [asr_datamodule.py:253] About to get test cuts
    2023-08-16 20:45:50,386 INFO [decode.py:203] batch 0/?, cuts processed until now is 4
    2023-08-16 20:45:50,556 INFO [decode.py:240] The transcripts are stored in tdnn/exp/recogs-test_set.txt
    2023-08-16 20:45:50,557 INFO [utils.py:564] [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
    2023-08-16 20:45:50,558 INFO [decode.py:248] Wrote detailed error stats to tdnn/exp/errs-test_set.txt
    2023-08-16 20:45:50,559 INFO [decode.py:315] Done!

We can see that it produces an identical WER as before.

We can also use it to decode files with the following command:

.. code-block:: bash

  # ./tdnn/pretrained.py requires kaldifeat
  #
  # Please refer to https://csukuangfj.github.io/kaldifeat/installation/from_wheels.html
  # for how to install kaldifeat

  pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.0 -f https://csukuangfj.github.io/kaldifeat/cpu.html

  ./tdnn/pretrained.py \
    --checkpoint ./tdnn/exp/pretrained.pt \
    --HLG ./data/lang_phone/HLG.pt \
    --words-file ./data/lang_phone/words.txt \
    download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    download/waves_yesno/0_0_1_0_0_0_1_0.wav

The output is given below:

.. code-block:: bash

  2023-08-16 20:53:19,208 INFO [pretrained.py:136] {'feature_dim': 23, 'num_classes': 4, 'sample_rate': 8000, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'checkpoint': './tdnn/exp/pretrained.pt', 'words_file': './data/lang_phone/words.txt', 'HLG': './data/lang_phone/HLG.pt', 'sound_files': ['download/waves_yesno/0_0_0_1_0_0_0_1.wav', 'download/waves_yesno/0_0_1_0_0_0_1_0.wav']}
  2023-08-16 20:53:19,208 INFO [pretrained.py:142] device: cpu
  2023-08-16 20:53:19,208 INFO [pretrained.py:144] Creating model
  2023-08-16 20:53:19,212 INFO [pretrained.py:156] Loading HLG from ./data/lang_phone/HLG.pt
  2023-08-16 20:53:19,213 INFO [pretrained.py:160] Constructing Fbank computer
  2023-08-16 20:53:19,213 INFO [pretrained.py:170] Reading sound files: ['download/waves_yesno/0_0_0_1_0_0_0_1.wav', 'download/waves_yesno/0_0_1_0_0_0_1_0.wav']
  2023-08-16 20:53:19,224 INFO [pretrained.py:176] Decoding started
  2023-08-16 20:53:19,304 INFO [pretrained.py:212]
  download/waves_yesno/0_0_0_1_0_0_0_1.wav:
  NO NO NO YES NO NO NO YES

  download/waves_yesno/0_0_1_0_0_0_1_0.wav:
  NO NO YES NO NO NO YES NO


  2023-08-16 20:53:19,304 INFO [pretrained.py:214] Decoding Done


Export via torch.jit.script()
-----------------------------

The command for this kind of export is

.. code-block:: bash

   cd /tmp/icefall
   export PYTHONPATH=/tmp/icefall:$PYTHONPATH
   cd egs/yesno/ASR

   # assume that "--epoch 14 --avg 2" produces the lowest WER.

   ./tdnn/export.py --epoch 14 --avg 2 --jit true

The output logs are given below:

.. code-block:: bash

  2023-08-16 20:47:44,666 INFO [export.py:76] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01, 'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, 'best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'reset_interval': 20, 'valid_interval': 10, 'beam_size': 10, 'reduction': 'sum', 'use_double_scores': True, 'epoch': 14, 'avg': 2, 'jit': True}
  2023-08-16 20:47:44,667 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
  2023-08-16 20:47:44,670 INFO [export.py:93] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
  2023-08-16 20:47:44,677 INFO [export.py:100] Using torch.jit.script
  2023-08-16 20:47:44,843 INFO [export.py:104] Saved to tdnn/exp/cpu_jit.pt

From the output logs we can see that the generated file is saved to ``tdnn/exp/cpu_jit.pt``.

Don't be confused by the name ``cpu_jit.pt``. The ``cpu`` part means the model is moved to
CPU before exporting. That means, when you load it with:

.. code-block:: bash

   torch.jit.load()

you don't need to specify the argument `map_location <https://pytorch.org/docs/stable/generated/torch.jit.load.html#torch.jit.load>`_
and it resides on CPU by default.

To use ``tdnn/exp/cpu_jit.pt`` with `icefall`_ to decode files, we can use:

.. code-block:: bash

  # ./tdnn/jit_pretrained.py requires kaldifeat
  #
  # Please refer to https://csukuangfj.github.io/kaldifeat/installation/from_wheels.html
  # for how to install kaldifeat

  pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.0 -f https://csukuangfj.github.io/kaldifeat/cpu.html


  ./tdnn/jit_pretrained.py \
    --nn-model ./tdnn/exp/cpu_jit.pt \
    --HLG ./data/lang_phone/HLG.pt \
    --words-file ./data/lang_phone/words.txt \
    download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    download/waves_yesno/0_0_1_0_0_0_1_0.wav

The output is given below:

.. code-block:: bash

  2023-08-16 20:56:00,603 INFO [jit_pretrained.py:121] {'feature_dim': 23, 'num_classes': 4, 'sample_rate': 8000, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'nn_model': './tdnn/exp/cpu_jit.pt', 'words_file': './data/lang_phone/words.txt', 'HLG': './data/lang_phone/HLG.pt', 'sound_files': ['download/waves_yesno/0_0_0_1_0_0_0_1.wav', 'download/waves_yesno/0_0_1_0_0_0_1_0.wav']}
  2023-08-16 20:56:00,603 INFO [jit_pretrained.py:127] device: cpu
  2023-08-16 20:56:00,603 INFO [jit_pretrained.py:129] Loading torchscript model
  2023-08-16 20:56:00,640 INFO [jit_pretrained.py:134] Loading HLG from ./data/lang_phone/HLG.pt
  2023-08-16 20:56:00,641 INFO [jit_pretrained.py:138] Constructing Fbank computer
  2023-08-16 20:56:00,641 INFO [jit_pretrained.py:148] Reading sound files: ['download/waves_yesno/0_0_0_1_0_0_0_1.wav', 'download/waves_yesno/0_0_1_0_0_0_1_0.wav']
  2023-08-16 20:56:00,642 INFO [jit_pretrained.py:154] Decoding started
  2023-08-16 20:56:00,727 INFO [jit_pretrained.py:190]
  download/waves_yesno/0_0_0_1_0_0_0_1.wav:
  NO NO NO YES NO NO NO YES

  download/waves_yesno/0_0_1_0_0_0_1_0.wav:
  NO NO YES NO NO NO YES NO


  2023-08-16 20:56:00,727 INFO [jit_pretrained.py:192] Decoding Done

.. hint::

   We provide only code for ``torch.jit.script()``. You can try ``torch.jit.trace()``
   if you want.

Export via torch.onnx.export()
------------------------------

The command for this kind of export is

.. code-block:: bash

   cd /tmp/icefall
   export PYTHONPATH=/tmp/icefall:$PYTHONPATH
   cd egs/yesno/ASR

   # tdnn/export_onnx.py requires onnx and onnxruntime
   pip install onnx onnxruntime

   # assume that "--epoch 14 --avg 2" produces the lowest WER.

   ./tdnn/export_onnx.py \
     --epoch 14 \
     --avg 2

The output logs are given below:

.. code-block:: bash

  2023-08-16 20:59:20,888 INFO [export_onnx.py:83] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01, 'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, 'best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'reset_interval': 20, 'valid_interval': 10, 'beam_size': 10, 'reduction': 'sum', 'use_double_scores': True, 'epoch': 14, 'avg': 2}
  2023-08-16 20:59:20,888 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
  2023-08-16 20:59:20,892 INFO [export_onnx.py:100] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
  ================ Diagnostic Run torch.onnx.export version 2.0.0 ================
  verbose: False, log level: Level.ERROR
  ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

  2023-08-16 20:59:21,047 INFO [export_onnx.py:127] Saved to tdnn/exp/model-epoch-14-avg-2.onnx
  2023-08-16 20:59:21,047 INFO [export_onnx.py:136] meta_data: {'model_type': 'tdnn', 'version': '1', 'model_author': 'k2-fsa', 'comment': 'non-streaming tdnn for the yesno recipe', 'vocab_size': 4}
  2023-08-16 20:59:21,049 INFO [export_onnx.py:140] Generate int8 quantization models
  2023-08-16 20:59:21,075 INFO [onnx_quantizer.py:538] Quantization parameters for tensor:"/Transpose_1_output_0" not specified
  2023-08-16 20:59:21,081 INFO [export_onnx.py:151] Saved to tdnn/exp/model-epoch-14-avg-2.int8.onnx

We can see from the logs that it generates two files:

  - ``tdnn/exp/model-epoch-14-avg-2.onnx`` (ONNX model with ``float32`` weights)
  - ``tdnn/exp/model-epoch-14-avg-2.int8.onnx`` (ONNX model with ``int8`` weights)

To use the generated ONNX model files for decoding with `onnxruntime`_, we can use

.. code-block:: bash

  # ./tdnn/onnx_pretrained.py requires kaldifeat
  #
  # Please refer to https://csukuangfj.github.io/kaldifeat/installation/from_wheels.html
  # for how to install kaldifeat

  pip install kaldifeat==1.25.3.dev20231221+cpu.torch2.0.0 -f https://csukuangfj.github.io/kaldifeat/cpu.html

  ./tdnn/onnx_pretrained.py \
    --nn-model ./tdnn/exp/model-epoch-14-avg-2.onnx \
    --HLG ./data/lang_phone/HLG.pt \
    --words-file ./data/lang_phone/words.txt \
    download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    download/waves_yesno/0_0_1_0_0_0_1_0.wav

The output is given below:

.. code-block:: bash

  2023-08-16 21:03:24,260 INFO [onnx_pretrained.py:166] {'feature_dim': 23, 'sample_rate': 8000, 'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'nn_model': './tdnn/exp/model-epoch-14-avg-2.onnx', 'words_file': './data/lang_phone/words.txt', 'HLG': './data/lang_phone/HLG.pt', 'sound_files': ['download/waves_yesno/0_0_0_1_0_0_0_1.wav', 'download/waves_yesno/0_0_1_0_0_0_1_0.wav']}
  2023-08-16 21:03:24,260 INFO [onnx_pretrained.py:171] device: cpu
  2023-08-16 21:03:24,260 INFO [onnx_pretrained.py:173] Loading onnx model ./tdnn/exp/model-epoch-14-avg-2.onnx
  2023-08-16 21:03:24,267 INFO [onnx_pretrained.py:176] Loading HLG from ./data/lang_phone/HLG.pt
  2023-08-16 21:03:24,270 INFO [onnx_pretrained.py:180] Constructing Fbank computer
  2023-08-16 21:03:24,273 INFO [onnx_pretrained.py:190] Reading sound files: ['download/waves_yesno/0_0_0_1_0_0_0_1.wav', 'download/waves_yesno/0_0_1_0_0_0_1_0.wav']
  2023-08-16 21:03:24,279 INFO [onnx_pretrained.py:196] Decoding started
  2023-08-16 21:03:24,318 INFO [onnx_pretrained.py:232]
  download/waves_yesno/0_0_0_1_0_0_0_1.wav:
  NO NO NO YES NO NO NO YES

  download/waves_yesno/0_0_1_0_0_0_1_0.wav:
  NO NO YES NO NO NO YES NO


  2023-08-16 21:03:24,318 INFO [onnx_pretrained.py:234] Decoding Done

.. note::

   To use the ``int8`` ONNX model for decoding, please use:

   .. code-block:: bash

      ./tdnn/onnx_pretrained.py \
        --nn-model ./tdnn/exp/model-epoch-14-avg-2.onnx \
        --HLG ./data/lang_phone/HLG.pt \
        --words-file ./data/lang_phone/words.txt \
        download/waves_yesno/0_0_0_1_0_0_0_1.wav \
        download/waves_yesno/0_0_1_0_0_0_1_0.wav

For the more curious
--------------------

If you are wondering how to deploy the model without ``torch``, please
continue reading. We will show how to use `sherpa-onnx`_ to run the
exported ONNX models, which depends only on `onnxruntime`_ and does not
depend on ``torch``.

In this tutorial, we will only demonstrate the usage of `sherpa-onnx`_ with the
pre-trained model of the `yesno`_ recipe. There are also other two frameworks
available:

  - `sherpa`_. It works with torchscript models.
  - `sherpa-ncnn`_. It works with models exported using :ref:`icefall_export_to_ncnn` with `ncnn`_

Please see `<https://k2-fsa.github.io/sherpa/>`_ for further details.
