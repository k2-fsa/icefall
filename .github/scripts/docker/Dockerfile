ARG PYTHON_VERSION=3.8
FROM python:${PYTHON_VERSION}

ARG TORCHAUDIO_VERSION="0.13.0"
ARG TORCH_VERSION="1.13.0"
ARG K2_VERSION="1.24.4.dev20231220"
ARG KALDIFEAT_VERSION="1.25.3.dev20231221"

ARG _K2_VERSION="${K2_VERSION}+cpu.torch${TORCH_VERSION}"
ARG _KALDIFEAT_VERSION="${KALDIFEAT_VERSION}+cpu.torch${TORCH_VERSION}"

RUN apt-get update -y && \
    apt-get install -qq -y \
    cmake \
    ffmpeg \
    git \
    git-lfs \
    graphviz \
    less \
    tree \
    vim \
    && \
    apt-get clean && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists


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
      matplotlib \
      multi_quantization \
      numba \
      "numpy<2.0" \
      onnxoptimizer \
      onnxsim \
      onnx \
      onnxmltools \
      onnxruntime \
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
