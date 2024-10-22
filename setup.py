#!/usr/bin/env python3

from pathlib import Path

from setuptools import find_packages, setup

icefall_dir = Path(__file__).parent
install_requires = (icefall_dir / "requirements.txt").read_text().splitlines()

setup(
    name="biofall",
    version="1.0",
    python_requires=">=3.9.0",
    description="Biomarker processing with Lhotse. (Forked from the K2 project)",
    author="The k2 and Lhotse Development Team",
    license="Apache-2.0 License",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
