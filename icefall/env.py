# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                    Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import k2
import k2.version
import lhotse
import torch


def get_git_sha1():
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                subprocess.run(
                    ["git", "diff", "--shortstat"],
                    check=True,
                    stdout=subprocess.PIPE,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + "-dirty" if dirty_commit else git_commit + "-clean"
    except:  # noqa
        return None

    return git_commit


def get_git_date():
    try:
        git_date = (
            subprocess.run(
                ["git", "log", "-1", "--format=%ad", "--date=local"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:  # noqa
        return None

    return git_date


def get_git_branch_name():
    try:
        git_date = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:  # noqa
        return None

    return git_date


def get_env_info() -> Dict[str, Any]:
    """Get the environment information."""
    return {
        "k2-version": k2.version.__version__,
        "k2-build-type": k2.version.__build_type__,
        "k2-with-cuda": k2.with_cuda,
        "k2-git-sha1": k2.version.__git_sha1__,
        "k2-git-date": k2.version.__git_date__,
        "lhotse-version": lhotse.__version__,
        "torch-version": str(torch.__version__),
        "torch-cuda-available": torch.cuda.is_available(),
        "torch-cuda-version": torch.version.cuda,
        "python-version": sys.version[:3],
        "icefall-git-branch": get_git_branch_name(),
        "icefall-git-sha1": get_git_sha1(),
        "icefall-git-date": get_git_date(),
        "icefall-path": str(Path(__file__).resolve().parent.parent),
        "k2-path": str(Path(k2.__file__).resolve()),
        "lhotse-path": str(Path(lhotse.__file__).resolve()),
        "hostname": socket.gethostname(),
        "IP address": socket.gethostbyname(socket.gethostname()),
    }
