#!/usr/bin/env bash

# This script installs kaldifeat into the directory ~/tmp/kaldifeat
# which is cached by GitHub actions for later runs.

set -e

mkdir -p ~/tmp
cd ~/tmp
git clone https://github.com/csukuangfj/kaldifeat
cd kaldifeat
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2 _kaldifeat
