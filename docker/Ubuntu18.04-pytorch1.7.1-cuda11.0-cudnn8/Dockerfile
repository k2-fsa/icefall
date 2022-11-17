FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# ENV http_proxy=http://aaa.bbb.cc.net:8080 \
#	https_proxy=http://aaa.bbb.cc.net:8080

RUN rm /etc/apt/sources.list.d/cuda.list && \
	rm /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-key del 7fa2af80

# install normal source
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        zlib1g-dev \
        gfortran \
        ca-certificates \
        patch \
        ffmpeg \
        valgrind \
		libssl-dev \
	    vim \
		curl

# Add new keys and reupdate
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - && \
	curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
	echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
	echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
	rm -rf /var/lib/apt/lists/* && \
	mv /opt/conda/lib/libcufft.so.10 /opt/libcufft.so.10.bak && \
    mv /opt/conda/lib/libcurand.so.10 /opt/libcurand.so.10.bak && \
    mv /opt/conda/lib/libcublas.so.11 /opt/libcublas.so.11.bak && \
    mv /opt/conda/lib/libnvrtc.so.11.0 /opt/libnvrtc.so.11.1.bak && \
    # mv /opt/conda/lib/libnvToolsExt.so.1 /opt/libnvToolsExt.so.1.bak && \
    mv /opt/conda/lib/libcudart.so.11.0 /opt/libcudart.so.11.0.bak && \
	apt-get update && apt-get -y upgrade

# cmake
RUN wget -P /opt https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz && \
    cd /opt && \
    tar -zxvf cmake-3.18.0.tar.gz && \
    cd cmake-3.18.0 && \
    ./bootstrap && \
    make && \
    make install && \
    rm -rf cmake-3.18.0.tar.gz && \
    find /opt/cmake-3.18.0 -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    cd -

# flac
RUN wget -P /opt https://downloads.xiph.org/releases/flac/flac-1.3.2.tar.xz  && \
    cd /opt && \
    xz -d flac-1.3.2.tar.xz && \
    tar -xvf flac-1.3.2.tar && \
    cd flac-1.3.2 && \
    ./configure && \
    make && make install && \
    rm -rf flac-1.3.2.tar && \
    find /opt/flac-1.3.2  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    cd -

RUN conda install -y -c pytorch torchaudio=0.7.1 && \
    pip install graphviz

#install k2 from source
RUN git clone https://github.com/k2-fsa/k2.git /opt/k2 && \
    cd /opt/k2 && \
    python3 setup.py install && \
    cd -

# install  lhotse
RUN pip install git+https://github.com/lhotse-speech/lhotse

RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
	cd /workspace/icefall && \
	pip install -r requirements.txt

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

WORKDIR /workspace/icefall
