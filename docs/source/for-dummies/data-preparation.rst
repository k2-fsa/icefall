.. _dummies_tutorial_data_preparation:

Data Preparation
================

After :ref:`dummies_tutorial_environment_setup`, we can start preparing the
data for training and decoding.

The first step is to prepare the data for training. We have already provided
`prepare.sh <https://github.com/k2-fsa/icefall/blob/master/egs/yesno/ASR/prepare.sh>`_
that would prepare everything required for training.

.. code-block::

   cd /tmp/icefall
   export PYTHONPATH=/tmp/icefall:$PYTHONPATH
   cd egs/yesno/ASR

   ./prepare.sh

Note that in each recipe from `icefall`_, there exists a file ``prepare.sh``,
which you should run before you run anything else.

That is all you need for data preparation.

For the more curious
--------------------

If you are wondering how to prepare your own dataset, please refer to the following
URLs for more details:

  - `<https://github.com/lhotse-speech/lhotse/tree/master/lhotse/recipes>`_

    It contains recipes for a variety of dataset. If you want to add your own
    dataset, please read recipes in this folder first.

  - `<https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/yesno.py>`_

    The `yesno`_ recipe in `lhotse`_.

If you already have a `Kaldi`_ dataset directory, which contains files like
``wav.scp``, ``feats.scp``, then you can refer to `<https://lhotse.readthedocs.io/en/latest/kaldi.html#example>`_.

A quick look to the generated files
-----------------------------------

``./prepare.sh`` puts generated files into two directories:

  - ``download``
  - ``data``

download
^^^^^^^^

The ``download`` directory contains downloaded dataset files:

.. code-block:: bas

    tree -L 1 ./download/

    ./download/
    |-- waves_yesno
    `-- waves_yesno.tar.gz

.. hint::

   Please refer to `<https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/yesno.py#L41>`_
   for how the data is downloaded and extracted.

data
^^^^

.. code-block:: bash

    tree ./data/

    ./data/
    |-- fbank
    |   |-- yesno_cuts_test.jsonl.gz
    |   |-- yesno_cuts_train.jsonl.gz
    |   |-- yesno_feats_test.lca
    |   `-- yesno_feats_train.lca
    |-- lang_phone
    |   |-- HLG.pt
    |   |-- L.pt
    |   |-- L_disambig.pt
    |   |-- Linv.pt
    |   |-- lexicon.txt
    |   |-- lexicon_disambig.txt
    |   |-- tokens.txt
    |   `-- words.txt
    |-- lm
    |   |-- G.arpa
    |   `-- G.fst.txt
    `-- manifests
        |-- yesno_recordings_test.jsonl.gz
        |-- yesno_recordings_train.jsonl.gz
        |-- yesno_supervisions_test.jsonl.gz
        `-- yesno_supervisions_train.jsonl.gz

    4 directories, 18 files

**data/manifests**:

  This directory contains manifests. They are used to generate files in
  ``data/fbank``.

  To give you an idea of what it contains, we examine the first few lines of
  the manifests related to the ``train`` dataset.

  .. code-block:: bash

      cd data/manifests
      gunzip -c  yesno_recordings_train.jsonl.gz  | head -n 3

  The output is given below:

    .. code-block:: bash

      {"id": "0_0_0_0_1_1_1_1", "sources": [{"type": "file", "channels": [0], "source": "/tmp/icefall/egs/yesno/ASR/download/waves_yesno/0_0_0_0_1_1_1_1.wav"}], "sampling_rate": 8000, "num_samples": 50800, "duration": 6.35, "channel_ids": [0]}
      {"id": "0_0_0_1_0_1_1_0", "sources": [{"type": "file", "channels": [0], "source": "/tmp/icefall/egs/yesno/ASR/download/waves_yesno/0_0_0_1_0_1_1_0.wav"}], "sampling_rate": 8000, "num_samples": 48880, "duration": 6.11, "channel_ids": [0]}
      {"id": "0_0_1_0_0_1_1_0", "sources": [{"type": "file", "channels": [0], "source": "/tmp/icefall/egs/yesno/ASR/download/waves_yesno/0_0_1_0_0_1_1_0.wav"}], "sampling_rate": 8000, "num_samples": 48160, "duration": 6.02, "channel_ids": [0]}

  Please refer to `<https://github.com/lhotse-speech/lhotse/blob/master/lhotse/audio.py#L300>`_
  for the meaning of each field per line.

  .. code-block:: bash

      gunzip -c  yesno_supervisions_train.jsonl.gz  | head -n 3

  The output is given below:

  .. code-block:: bash

      {"id": "0_0_0_0_1_1_1_1", "recording_id": "0_0_0_0_1_1_1_1", "start": 0.0, "duration": 6.35, "channel": 0, "text": "NO NO NO NO YES YES YES YES", "language": "Hebrew"}
      {"id": "0_0_0_1_0_1_1_0", "recording_id": "0_0_0_1_0_1_1_0", "start": 0.0, "duration": 6.11, "channel": 0, "text": "NO NO NO YES NO YES YES NO", "language": "Hebrew"}
      {"id": "0_0_1_0_0_1_1_0", "recording_id": "0_0_1_0_0_1_1_0", "start": 0.0, "duration": 6.02, "channel": 0, "text": "NO NO YES NO NO YES YES NO", "language": "Hebrew"}

  Please refer to `<https://github.com/lhotse-speech/lhotse/blob/master/lhotse/supervision.py#L510>`_
  for the meaning of each field per line.

**data/fbank**:

  This directory contains everything from ``data/manifests``. Furthermore, it also contains features
  for training.

  ``data/fbank/yesno_feats_train.lca`` contains the features for the train dataset.
  Features are compressed using `lilcom`_.

  ``data/fbank/yesno_cuts_train.jsonl.gz`` stores the `CutSet <https://github.com/lhotse-speech/lhotse/blob/master/lhotse/cut/set.py#L72>`_,
  which stores `RecordingSet <https://github.com/lhotse-speech/lhotse/blob/master/lhotse/audio.py#L928>`_,
  `SupervisionSet <https://github.com/lhotse-speech/lhotse/blob/master/lhotse/supervision.py#L510>`_,
  and `FeatureSet <https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/base.py#L593>`_.

  To give you an idea about what it looks like, we can run the following command:

    .. code-block:: bash

        cd data/fbank

        gunzip -c yesno_cuts_train.jsonl.gz | head -n 3

  The output is given below:

    .. code-block:: bash

      {"id": "0_0_0_0_1_1_1_1-0", "start": 0, "duration": 6.35, "channel": 0, "supervisions": [{"id": "0_0_0_0_1_1_1_1", "recording_id": "0_0_0_0_1_1_1_1", "start": 0.0, "duration": 6.35, "channel": 0, "text": "NO NO NO NO YES YES YES YES", "language": "Hebrew"}], "features": {"type": "kaldi-fbank", "num_frames": 635, "num_features": 23, "frame_shift": 0.01, "sampling_rate": 8000, "start": 0, "duration": 6.35, "storage_type": "lilcom_chunky", "storage_path": "data/fbank/yesno_feats_train.lca", "storage_key": "0,13000,3570", "channels": 0}, "recording": {"id": "0_0_0_0_1_1_1_1", "sources": [{"type": "file", "channels": [0], "source": "/tmp/icefall/egs/yesno/ASR/download/waves_yesno/0_0_0_0_1_1_1_1.wav"}], "sampling_rate": 8000, "num_samples": 50800, "duration": 6.35, "channel_ids": [0]}, "type": "MonoCut"}
      {"id": "0_0_0_1_0_1_1_0-1", "start": 0, "duration": 6.11, "channel": 0, "supervisions": [{"id": "0_0_0_1_0_1_1_0", "recording_id": "0_0_0_1_0_1_1_0", "start": 0.0, "duration": 6.11, "channel": 0, "text": "NO NO NO YES NO YES YES NO", "language": "Hebrew"}], "features": {"type": "kaldi-fbank", "num_frames": 611, "num_features": 23, "frame_shift": 0.01, "sampling_rate": 8000, "start": 0, "duration": 6.11, "storage_type": "lilcom_chunky", "storage_path": "data/fbank/yesno_feats_train.lca", "storage_key": "16570,12964,2929", "channels": 0}, "recording": {"id": "0_0_0_1_0_1_1_0", "sources": [{"type": "file", "channels": [0], "source": "/tmp/icefall/egs/yesno/ASR/download/waves_yesno/0_0_0_1_0_1_1_0.wav"}], "sampling_rate": 8000, "num_samples": 48880, "duration": 6.11, "channel_ids": [0]}, "type": "MonoCut"}
      {"id": "0_0_1_0_0_1_1_0-2", "start": 0, "duration": 6.02, "channel": 0, "supervisions": [{"id": "0_0_1_0_0_1_1_0", "recording_id": "0_0_1_0_0_1_1_0", "start": 0.0, "duration": 6.02, "channel": 0, "text": "NO NO YES NO NO YES YES NO", "language": "Hebrew"}], "features": {"type": "kaldi-fbank", "num_frames": 602, "num_features": 23, "frame_shift": 0.01, "sampling_rate": 8000, "start": 0, "duration": 6.02, "storage_type": "lilcom_chunky", "storage_path": "data/fbank/yesno_feats_train.lca", "storage_key": "32463,12936,2696", "channels": 0}, "recording": {"id": "0_0_1_0_0_1_1_0", "sources": [{"type": "file", "channels": [0], "source": "/tmp/icefall/egs/yesno/ASR/download/waves_yesno/0_0_1_0_0_1_1_0.wav"}], "sampling_rate": 8000, "num_samples": 48160, "duration": 6.02, "channel_ids": [0]}, "type": "MonoCut"}

  Note that ``yesno_cuts_train.jsonl.gz`` only stores the information about how to read the features.
  The actual features are stored separately in ``data/fbank/yesno_feats_train.lca``.

**data/lang**:

  This directory contains the lexicon.

**data/lm**:

  This directory contains language models.
