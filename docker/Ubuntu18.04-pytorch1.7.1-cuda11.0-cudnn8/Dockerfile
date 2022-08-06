FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

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
	    vim && \
        rm -rf /var/lib/apt/lists/*


RUN mv /opt/conda/lib/libcufft.so.10 /opt/libcufft.so.10.bak && \
    mv /opt/conda/lib/libcurand.so.10 /opt/libcurand.so.10.bak && \
    mv /opt/conda/lib/libcublas.so.11 /opt/libcublas.so.11.bak && \
    mv /opt/conda/lib/libnvrtc.so.11.0 /opt/libnvrtc.so.11.1.bak && \
    mv /opt/conda/lib/libnvToolsExt.so.1 /opt/libnvToolsExt.so.1.bak && \
    mv /opt/conda/lib/libcudart.so.11.0 /opt/libcudart.so.11.0.bak

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

#kaldiio

RUN pip install kaldiio

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

# graphviz
RUN pip install graphviz

# kaldifeat
RUN git clone https://github.com/csukuangfj/kaldifeat.git /opt/kaldifeat && \
    cd /opt/kaldifeat && \
    python setup.py install && \
    cd -


#install k2 from source
RUN git clone https://github.com/k2-fsa/k2.git /opt/k2 && \
    cd /opt/k2 && \
    python3 setup.py install && \
    cd -

# install  lhotse
RUN pip install torchaudio==0.7.2
RUN pip install git+https://github.com/lhotse-speech/lhotse
#RUN pip install lhotse

# install icefall
RUN git clone https://github.com/k2-fsa/icefall && \
    cd icefall && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
ENV PYTHONPATH /workspace/icefall:$PYTHONPATH  

