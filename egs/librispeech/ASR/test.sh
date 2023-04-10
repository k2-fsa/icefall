git pull
spk_id=$1
#for i in 10 20 30 40 50 60 70 80 90 100
for i in 100
do
	for method in modified_beam_search
	do
	  ./pruned_transducer_stateless_d2v_v2/decode.py \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--additional-block True \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/"$spk_id"_adapter_10m \
		--model-name checkpoint-$i.pt \
		--max-duration 400 \
		--decoding-method $method \
		--max-sym-per-frame 1 \
		--encoder-type d2v \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--avg 1 \
		--use-averaged-model True \
		--spk-id $spk_id \
		--res-name beam_adapter_RA_spec_10m_$i
	done
done

#--model-name epoch-$i.pt \
#--model-name ../d2v-base-T.pt \
