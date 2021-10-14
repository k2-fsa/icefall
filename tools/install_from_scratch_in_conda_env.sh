#!/usr/bin/env bash

set -eou pipefail  # "strict" mode
set -x  # Print executed messages

PYTHON_VER=3.8
PYTORCH_VER=1.8.1
CUDA_VER=11.1

echo Installing k2 and PyTorch
conda install -c k2-fsa -c pytorch -c conda-forge k2 python=$PYTHON_VER cudatoolkit=$CUDA_VER pytorch=$PYTORCH_VER

echo Installing Lhotse
pip install git+https://github.com/lhotse-speech/lhotse

echo "Installing cmake (avoids issues on older systems)"
pip install cmake

echo Installing Icefall
pip install -e .

echo Testing installation - imports
python -c "import k2; import lhotse; import icefall"

echo Testing installation - yesno recipe
cd egs/yesno/ASR
./prepare.sh
tdnn/train.py
tdnn/decode.py

echo "All set!"
