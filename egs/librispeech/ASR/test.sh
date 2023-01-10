git pull

#for method in greedy_search modified_beam_search fast_beam_search; do
for i in {1..6}
do
  ./pruned_transducer_stateless_d2v_v2/decode.py \
	--input-strategy AudioSamples \
	--enable-spec-aug False \
	--additional-block True \
	--model-name epoch-$i.pt \
	--exp-dir ./pruned_transducer_stateless_d2v_v2/male_hpo6 \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
	--encoder-type d2v \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768 
done
