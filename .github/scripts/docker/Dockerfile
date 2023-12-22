FROM k2fsa/icefall:torch1.13.0-cuda11.6

WORKDIR /workspace/icefall

RUN cd egs/librispeech/ASR && \
    ./prepare.sh --stop-stage 1 && \
    ls -lh download data
