
#export CUDA_VISIBLE_DEVICES="2,3"
pip install -r seamlessm4t/requirements.txt
pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/sa/yuekaiz/asr/icefall
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/sa/yuekaiz/asr/seamless_communication/src
export TORCH_HOME=/lustre/fsw/sa/yuekaiz/asr/hub
torchrun --nproc-per-node 8 seamlessm4t/train2.py --use-fp16 1 --max-duration 300 --base-lr 1e-5 --exp-dir seamlessm4t/exp
