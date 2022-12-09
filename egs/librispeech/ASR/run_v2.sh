export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

git pull

./pruned_transducer_stateless_d2v_v2/train.py \
	--input-strategy AudioSamples \
	--enable-spec-aug False \
	--world-size 8 \
	--num-epochs 30 \
	--full-libri 1 \
	--use-fp16 1 \
	--max-duration 300 \
	--exp-dir pruned_transducer_stateless_d2v_v2/exp \
	--encoder-dims "768 768 768 768 768" \
	--feedforward-dims  "1024,1024,2048,2048,1024" \
	--ctc-loss-scale 0.2 \
	--master-port 12535
