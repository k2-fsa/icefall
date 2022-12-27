git pull

workstation="bear"

if [ $workstation = "whale" ]; then
	export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
	./pruned_transducer_stateless_d2v_v2/train.py \
		--add-adapter True \
		--adapter-lr 0.000025 \
		--wandb False \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--multi-optim False \
		--world-size 8 \
		--num-epochs 10 \
		--start-epoch 1 \
		--full-libri 1 \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
		--max-duration 200 \
		--use-fp16 0 \
		--encoder-type d2v \
		--additional-block True \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--prune-range 10 
else
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
	./pruned_transducer_stateless_d2v_v2/train.py \
		--wandb False \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--multi-optim False \
		--start-epoch 28 \
		--world-size 4 \
		--num-epochs 30 \
		--full-libri 1 \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
		--max-duration 150 \
		--freeze-finetune-updates 3000 \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--use-fp16 0 \
		--peak-dec-lr 0.04175 \
		--peak-enc-lr 0.0003859 \
		--accum-grads 4 \
		--encoder-type d2v \
		--additional-block True \
		--prune-range 10 \
		--context-size 2 \
		--ctc-loss-scale 0.2 
fi
