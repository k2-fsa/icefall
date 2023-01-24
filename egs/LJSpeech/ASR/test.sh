git pull

for method in greedy_search modified_beam_search fast_beam_search
do
  ./pruned_transducer_stateless_d2v_v2/decode.py \
	--manifest-dir data/fbank_full \
	--input-strategy AudioSamples \
	--enable-spec-aug False \
	--additional-block True \
	--model-name epoch-15.pt \
	--exp-dir ./pruned_transducer_stateless_d2v_v2/adapter_60 \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
	--encoder-type d2v \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768 \
	--avg 1 \
	--use-averaged-model True
done
