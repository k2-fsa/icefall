
#export CUDA_VISIBLE_DEVICES="2,3"
#pip install -r seamlessm4t/requirements.txt
#pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/sa/yuekaiz/asr/icefall
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/sa/yuekaiz/asr/seamless_communication/src
export TORCH_HOME=/lustre/fsw/sa/yuekaiz/asr/hub
python3 seamlessm4t/decode.py --epoch 3 --exp-dir seamlessm4t/exp
