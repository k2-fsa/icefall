# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "icefall"
copyright = "2021, icefall development team"
author = "icefall development team"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "sphinxcontrib.youtube",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
}
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "installation/images"]

pygments_style = "sphinx"

numfig = True

html_context = {
    "display_github": True,
    "github_user": "k2-fsa",
    "github_repo": "icefall",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}

todo_include_todos = True

rst_epilog = """
.. _sherpa-ncnn: https://github.com/k2-fsa/sherpa-ncnn
.. _sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
.. _icefall: https://github.com/k2-fsa/icefall
.. _git-lfs: https://git-lfs.com/
.. _ncnn: https://github.com/tencent/ncnn
.. _LibriSpeech: https://www.openslr.org/12
.. _Gigaspeech: https://github.com/SpeechColab/GigaSpeech
.. _musan: http://www.openslr.org/17/
.. _ONNX: https://github.com/onnx/onnx
.. _onnxruntime: https://github.com/microsoft/onnxruntime
.. _torch: https://github.com/pytorch/pytorch
.. _torchaudio: https://github.com/pytorch/audio
.. _k2: https://github.com/k2-fsa/k2
.. _lhotse: https://github.com/lhotse-speech/lhotse
.. _yesno: https://www.openslr.org/1/
.. _Next-gen Kaldi: https://github.com/k2-fsa
.. _Kaldi: https://github.com/kaldi-asr/kaldi
.. _lilcom: https://github.com/danpovey/lilcom
"""
