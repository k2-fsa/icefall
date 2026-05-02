#!/usr/bin/env bash
set -euo pipefail

stage=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ICEFALL_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
AVHUBERT_DIR="$SCRIPT_DIR/av_hubert"

. "$ICEFALL_DIR/shared/parse_options.sh" || exit 1

k2_wheel="k2-1.24.4.dev20241030+cuda12.1.torch2.4.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

if [ $stage -le 1 ] ; then

  if  [ ! -f "$k2_wheel" ]; then
    wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/1.24.4.dev20241029/$k2_wheel
  fi
  # install k2
  pip install "$k2_wheel"
  # install lhotse
  pip install git+https://github.com/lhotse-speech/lhotse
  # install icefall requirements
  pip install -r "$ICEFALL_DIR/requirements.txt"
fi

if [ $stage -le 2 ] ; then
  if [ ! -d "$AVHUBERT_DIR" ]; then
    git clone https://github.com/facebookresearch/av_hubert.git  "$AVHUBERT_DIR"
  fi

  cd "$AVHUBERT_DIR"
  git submodule init
  git submodule update

  cd fairseq
  pip install --editable ./
fi

if [ $stage -le 3 ] ; then
  cd $ICEFALL_DIR/egs/grid/VSR
  
  pip install "pip<24.1"
  pip install -r grid-requirements.txt
  conda install -y -c conda-forge dlib==19.18.0 
  pip install  --force-reinstall --no-deps numpy==1.23.5

  echo 'export PYTHONPATH=$ICEFALL_DIR:$PYTHONPATH'
  echo 'Add the line above to your ~/.bashrc or conda activate hook'
fi
