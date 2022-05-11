FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# install normal source

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

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


RUN conda install pytorch torchvision torchaudio=0.11 cudatoolkit=11.3 -c pytorch



#install k2 from source
# RUN conda install -c k2-fsa -c pytorch -c conda-forge k2 cudatoolkit=11.3 pytorch=1.10.0
RUN git clone https://github.com/k2-fsa/k2.git /opt/k2 && \
    cd /opt/k2 && \
    python3 setup.py install && \
    cd -
# RUN pip install k2

# install  lhotse
# RUN pip install git+https://github.com/lhotse-speech/lhotse
RUN pip install lhotse

# install icefall
# RUN git clone https://github.com/k2-fsa/icefall && \
#     cd icefall && \
#     pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH  

