#!/usr/bin/env python3
"""Run Lhotse's Click CLI when its console-script entry point is absent.

Some CUDA-focused Icefall images retain the ``lhotse`` Python package but omit
the setuptools console-script wrapper.  Importing ``lhotse.bin.lhotse``
registers all command groups, then invoking ``cli`` reproduces ``lhotse ...``.
"""

# Importing this module registers all of Lhotse's Click command groups.
from lhotse.bin.lhotse import *  # noqa: F401,F403
from lhotse.bin.modes.cli_base import cli


if __name__ == "__main__":
    cli()
