

pip install -r whisper/requirements.txt
pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
export PYTHONPATH=$PYTHONPATH:/mnt/samsung-t7/yuekai/asr/icefall

torchrun --nproc-per-node 8 whisper/train.py --use-fp16 1 --max-duration 20 --base-lr 1e-5 --exp-dir whisper/exp_medimum --start-epoch 1
