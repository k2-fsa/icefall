Contributing to Documentation
=============================

We use `sphinx <https://www.sphinx-doc.org/en/master/>`_
for documentation.

Before writing documentation, you have to prepare the environment:

  .. code-block:: bash

    $ cd docs
    $ pip install -r requirements.txt

After setting up the environment, you are ready to write documentation.
Please refer to `reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
if you are not familiar with ``reStructuredText``.

After writing some documentation, you can build the documentation **locally**
to preview what it looks like if it is published:

  .. code-block:: bash

    $ cd docs
    $ make html

The generated documentation is in ``docs/build/html`` and can be viewed
with the following commands:

  .. code-block:: bash

    $ cd docs/build/html
    $ python3 -m http.server

It will print::

  Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...

Open your browser, go to `<http://0.0.0.0:8000/>`_, and you will see
the following:

    .. figure:: images/doc-contrib.png
       :width: 600
       :align: center

       View generated documentation locally with ``python3 -m http.server``.
