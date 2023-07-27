FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

LABEL authors="Fangjun Kuang"
LABEL k2_version="1.24.3.dev20230725+cuda11.6.torch1.13.0"
LABEL kaldifeat_version="1.25.0.dev20230726+cuda11.6.torch1.13.0"
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
      torchaudio==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html \
      k2==1.24.3.dev20230725+cuda11.6.torch1.13.0 -f https://k2-fsa.github.io/k2/cuda.html \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==1.25.0.dev20230726+cuda11.6.torch1.13.0 -f https://csukuangfj.github.io/kaldifeat/cuda.html \
      \
      kaldifst \
      kaldilm \
      kaldialign \
      sentencepiece>=0.1.96 \
      tensorboard \
      typeguard \
      dill

RUN cd /opt/conda/lib/stubs && ln -s libcuda.so libcuda.so.1

RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
	cd /workspace/icefall && \
	pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

ENV LD_LIBRARY_PATH /opt/conda/lib/stubs:$LD_LIBRARY_PATH

WORKDIR /workspace/icefall

