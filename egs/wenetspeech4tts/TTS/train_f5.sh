export PYTHONPATH=$PYTHONPATH:/home/yuekaiz/icefall_matcha

install_flag=false
if [ "$install_flag" = true ]; then
    echo "Installing packages..."

    pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
    # pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html
    # lhotse tensorboard kaldialign
    pip install -r requirements.txt
    pip install phonemizer pypinyin sentencepiece kaldialign matplotlib h5py

    apt-get update && apt-get -y install festival espeak-ng mbrola
else
    echo "Skipping installation."
fi

world_size=8
#world_size=1

exp_dir=exp/f5

# pip install -r f5-tts/requirements.txt
python3 f5-tts/train.py --max-duration 300 --filter-min-duration 0.5 --filter-max-duration 20  \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 8000 \
      --base-lr 1e-4 --warmup-steps 5000 --average-period 200 \
      --num-epochs 10 --start-epoch 1 --start-batch 20000 \
      --exp-dir ${exp_dir} --world-size ${world_size}
