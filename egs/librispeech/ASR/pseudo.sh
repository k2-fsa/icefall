spk_id=$1
prefix=$2

for method in modified_beam_search; do #modified_beam_search fast_beam_search; do
	  ./pruned_transducer_stateless_d2v_v2/pseudo.py \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--additional-block True \
		--model-name d2v-base-T.pt \
		--exp-dir ./pruned_transducer_stateless_d2v_v2 \
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
		--prefix $prefix
done

   #for spk_id in 1089 1188 121 1221 1284 1320 1580 1995 2094 2300 237 260 2830 2961 3570 3575 3729 4077 4446 4507 4970 4992 5105 5142 5639 5683 61 672 6829 6930 7021 7127 7176 7729 8224 8230 8455 8463 8555 908 1688 1998 2033 2414 2609 3005 3080 3331 3528 3538 367 3764 3997 4198 4294 4350 4852 533 5442 5484 5764 6070 6128 6432 6938 7018 7105 7902 7975 8131 8188 8280 8461; do 	
#done
#--model-name d2v-base-T.pt \
#--exp-dir ./pruned_transducer_stateless_d2v_v2 \

#--model-name epoch-30.pt \
#--exp-dir ./pruned_transducer_stateless_d2v_v2/"$spk_id"_adapter \

