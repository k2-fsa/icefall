FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# python 3.10

ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

# python 3.10
ARG K2_VERSION="1.24.4.dev20240223+cuda12.1.torch2.1.0"
ARG KALDIFEAT_VERSION="1.25.4.dev20240223+cuda12.1.torch2.1.0"
ARG TORCHAUDIO_VERSION="2.1.0+cu121"

LABEL authors="Fangjun Kuang <csukuangfj@gmail.com>"
LABEL k2_version=${K2_VERSION}
LABEL kaldifeat_version=${KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        vim \
    	libssl-dev \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        ffmpeg \
        g++ \
        gfortran \
        git \
        libtool \
        make \
        patch \
        sox \
        subversion \
        unzip \
        valgrind \
        wget \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
      torchaudio==${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/torch_stable.html \
      k2==${K2_VERSION} -f https://k2-fsa.github.io/k2/cuda.html \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==${KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cuda.html \
      kaldi_native_io \
      kaldialign \
      kaldifst \
      kaldilm \
      sentencepiece>=0.1.96 \
      tensorboard \
      typeguard \
      dill \
      onnx \
      onnxruntime \
      onnxmltools \
      multi_quantization \
      typeguard \
      numpy \
      pytest \
      graphviz

RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
    cd /workspace/icefall && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

WORKDIR /workspace/icefall
