FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

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

RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
	cd /workspace/icefall && \
	pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

WORKDIR /workspace/icefall

