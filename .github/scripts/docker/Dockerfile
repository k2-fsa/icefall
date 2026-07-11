ARG BASE_IMAGE=pytorch/manylinux2_28-builder:cpu
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=10
ARG TORCHAUDIO_VERSION="2.10.0"
ARG TORCH_VERSION="2.10.0"
ARG K2_VERSION="1.24.4.dev20260625"
ARG KALDIFEAT_VERSION="1.25.5.dev20260626"

ARG _K2_VERSION="${K2_VERSION}+cpu.torch${TORCH_VERSION}"
ARG _KALDIFEAT_VERSION="${KALDIFEAT_VERSION}+cpu.torch${TORCH_VERSION}"

# Set up Python from the manylinux image (/opt/python/cpXY-cpXY/)
ENV PYTHON_INSTALL_DIR=/opt/python/cp${PYTHON_MAJOR}${PYTHON_MINOR}-cp${PYTHON_MAJOR}${PYTHON_MINOR}
ENV PATH="${PYTHON_INSTALL_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONPATH="${PYTHON_INSTALL_DIR}/lib/python${PYTHON_VERSION}/site-packages"

RUN yum install -y epel-release && \
    yum install -y \
    cmake \
    ffmpeg \
    git \
    git-lfs \
    graphviz \
    less \
    tree \
    vim \
    && \
    yum clean all


LABEL authors="Fangjun Kuang <csukuangfj@gmail.com>"
LABEL k2_version=${_K2_VERSION}
LABEL kaldifeat_version=${_KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

# Install dependencies
RUN pip install --no-cache-dir \
      torch==${TORCH_VERSION}+cpu -f https://download.pytorch.org/whl/torch \
      torchaudio==${TORCHAUDIO_VERSION}+cpu -f https://download.pytorch.org/whl/torchaudio \
      k2==${_K2_VERSION} -f https://k2-fsa.github.io/k2/cpu.html \
      \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==${_KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cpu.html \
      conformer==0.3.2 \
      cython \
      diffusers \
      dill \
      espnet_tts_frontend \
      graphviz \
      kaldi-decoder \
      kaldi_native_io \
      kaldialign \
      kaldifst \
      kaldilm \
      librosa \
      "matplotlib<=3.9.4" \
      multi_quantization \
      numba \
      "numpy<2.0" \
      onnxoptimizer \
      onnxsim \
      onnx==1.17.0 \
      onnxmltools \
      onnxruntime==1.17.1 \
      piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html \
      pypinyin==0.50.0 \
      pytest \
      sentencepiece>=0.1.96 \
      six \
      tensorboard \
      typeguard

# RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
#     cd /workspace/icefall && \
#     pip install --no-cache-dir -r requirements.txt
#
# ENV PYTHONPATH /workspace/icefall:$PYTHONPATH
#
# WORKDIR /workspace/icefall
