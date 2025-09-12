FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

# python 3.7
ARG K2_VERSION="1.24.4.dev20240223+cuda10.2.torch1.9.0"
ARG KALDIFEAT_VERSION="1.25.4.dev20240223+cuda10.2.torch1.9.0"
ARG TORCHAUDIO_VERSION="0.9.0"

LABEL authors="Fangjun Kuang <csukuangfj@gmail.com>"
LABEL k2_version=${K2_VERSION}
LABEL kaldifeat_version=${KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

# see https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/

RUN rm /etc/apt/sources.list.d/cuda.list && \
	rm /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-key del 7fa2af80


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

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm -v cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip uninstall -y tqdm && \
    pip install -U --no-cache-dir \
      torchaudio==${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/torch_stable.html \
      k2==${K2_VERSION} -f https://k2-fsa.github.io/k2/cuda.html \
      kaldifeat==${KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cuda.html \
      git+https://github.com/lhotse-speech/lhotse \
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
      onnxoptimizer \
      onnxsim \
      multi_quantization \
      typeguard \
      numpy \
      pytest \
      graphviz \
      tqdm>=4.63.0


RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
    cd /workspace/icefall && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

WORKDIR /workspace/icefall
