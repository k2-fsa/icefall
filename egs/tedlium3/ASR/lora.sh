workstation=$3

if [ $workstation = "whale" ]; then
	#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
	export CUDA_VISIBLE_DEVICES=0
	if [ ! -e ./pruned_transducer_stateless_d2v_v2/$1/.train.done ]; then
		./pruned_transducer_stateless_d2v_v2/prompt_tuning.py \
			--num-buckets 2 \
			--add-adapter True \
			--adapter-lr 0.1 \
			--gender male \
			--wandb False \
			--input-strategy AudioSamples \
			--enable-spec-aug False \
			--multi-optim False \
			--world-size 1 \
			--num-epochs 10000 \
			--num-updates 101 \
			--save-every-n 50 \
			--full-libri 1 \
			--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
			--max-duration 200 \
			--encoder-dim 768 \
			--decoder-dim 768 \
			--joiner-dim 768 \
			--use-fp16 0 \
			--accum-grads 8 \
			--encoder-type d2v \
			--additional-block True \
			--prune-range 10 \
			--prompt True \
			--spk-id $2
		touch ./pruned_transducer_stateless_d2v_v2/$1/.train.done
	fi

else
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
	#rm ./pruned_transducer_stateless_d2v_v2/$1/.train.done
	if [ ! -e ./pruned_transducer_stateless_d2v_v2/$1/.train.done ]; then
		./pruned_transducer_stateless_d2v_v2/train_tta.py \
			--num-buckets 2 \
			--pea True \
			--lora True \
			--pea-lr 0.01 \
			--wandb False \
			--input-strategy AudioSamples \
			--enable-spec-aug False \
			--multi-optim False \
			--world-size 4 \
			--num-epochs 10000 \
			--num-updates 101 \
			--save-every-n 50 \
			--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
			--max-duration 150 \
			--accum-grads 2 \
			--encoder-dim 768 \
			--decoder-dim 768 \
			--joiner-dim 768 \
			--use-fp16 0 \
			--accum-grads 2 \
			--encoder-type d2v \
			--additional-block True \
			--prune-range 10 \
			--spk-id $2 \
			--ctc-loss-scale 0.2 \
			--bpe-model ./../../librispeech/ASR/data/lang_bpe_500/bpe.model
		#touch ./pruned_transducer_stateless_d2v_v2/$1/.train.done
	fi
fi
