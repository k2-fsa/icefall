
export CUDA_VISIBLE_DEVICES="2,3"
export PYTHONPATH=$PYTHONPATH:/mnt/samsung-t7/yuekai/asr/icefall
torchrun --nproc-per-node 2 seamlessm4t/train2.py --use-fp16 1 --max-duration 20
