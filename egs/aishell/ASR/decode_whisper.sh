
#export CUDA_VISIBLE_DEVICES="1"
#pip install -r whisper/requirements.txt
#pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/sa/yuekaiz/asr/icefall
#export PYTHONPATH=$PYTHONPATH:/mnt/samsung-t7/yuekai/asr/icefall/

python3 whisper/decode.py --exp-dir whisper/exp --max-duration 100
