git pull

for group_num in 2 3 4 6 12
do
	export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
	./pruned_transducer_stateless_gtrans/train.py \
				--world-size 8 \
				--num-epochs 30 \
				--start-epoch 1 \
				--full-libri 1 \
				--exp-dir pruned_transducer_stateless_gtrans/nocombine_layer12_group$group_num	\
				--max-duration 700 \
				--use-fp16 1 \
				--num-encoder-layers 12 \
				--group-num $group_num \
				--dim-feedforward 2048 \
				--nhead 8 \
				--encoder-dim 512 \
				--decoder-dim 512 \
				--joiner-dim 512
done
