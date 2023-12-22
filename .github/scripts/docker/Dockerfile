ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}

RUN apt-get update -y && \
    apt-get install -qq -y \
    git \
    git-lfs \
    less \
    vim \
    && \
    apt-get clean && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists

ARG K2_VERSION="1.24.4.dev20231220+cpu.torch1.13.0"
ARG KALDIFEAT_VERSION="1.25.3.dev20231221+cpu.torch1.13.0"
ARG TORCHAUDIO_VERSION="0.13.0"
ARG TORCH_VERSION="1.13.0"

LABEL authors="Fangjun Kuang <csukuangfj@gmail.com>"
LABEL k2_version=${K2_VERSION}
LABEL kaldifeat_version=${KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

# Install dependencies
RUN pip install --no-cache-dir \
      torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/cpu/torch_stable.html \
      k2==${K2_VERSION} -f https://k2-fsa.github.io/k2/cpu.html \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==${KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cpu.html \
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
      six \
      multi_quantization \
      typeguard \
      numpy \
      pytest \
      graphviz

# RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
#     cd /workspace/icefall && \
#     pip install --no-cache-dir -r requirements.txt
#
# ENV PYTHONPATH /workspace/icefall:$PYTHONPATH
#
# WORKDIR /workspace/icefall
