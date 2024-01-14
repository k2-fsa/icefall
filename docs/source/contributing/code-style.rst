.. _follow the code style:

Follow the code style
=====================

We use the following tools to make the code style to be as consistent as possible:

  - `black <https://github.com/psf/black>`_, to format the code
  - `flake8 <https://github.com/PyCQA/flake8>`_, to check the style and quality of the code
  - `isort <https://github.com/PyCQA/isort>`_, to sort ``imports``

The following versions of the above tools are used:

  - ``black == 22.3.0``
  - ``flake8 == 5.0.4``
  - ``isort == 5.10.1``

After running the following commands:

  .. code-block::

    $ git clone https://github.com/k2-fsa/icefall
    $ cd icefall
    $ pip install pre-commit
    $ pre-commit install

it will run the following checks whenever you run ``git commit``, **automatically**:

    .. figure:: images/pre-commit-check.png
       :width: 600
       :align: center

       pre-commit hooks invoked by ``git commit`` (Failed).

If any of the above checks failed, your ``git commit`` was not successful.
Please fix any issues reported by the check tools.

.. HINT::

  Some of the check tools, i.e., ``black`` and ``isort`` will modify
  the files to be committed **in-place**. So please run ``git status``
  after failure to see which file has been modified by the tools
  before you make any further changes.

After fixing all the failures, run ``git commit`` again and
it should succeed this time:

    .. figure:: images/pre-commit-check-success.png
       :width: 600
       :align: center

       pre-commit hooks invoked by ``git commit`` (Succeeded).

If you want to check the style of your code before ``git commit``, you
can do the following:

  .. code-block:: bash

    $ pre-commit install
    $ pre-commit run

Or without installing the pre-commit hooks:

  .. code-block:: bash

    $ cd icefall
    $ pip install black==22.3.0 flake8==5.0.4 isort==5.10.1
    $ black --check your_changed_file.py
    $ black your_changed_file.py  # modify it in-place
    $
    $ flake8 your_changed_file.py
    $
    $ isort --check your_changed_file.py  # modify it in-place
    $ isort your_changed_file.py
