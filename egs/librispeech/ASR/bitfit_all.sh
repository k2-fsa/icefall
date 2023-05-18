#dl_dir=/DB/LibriSpeech_tar/vox
subset=$1
dl_dir=/DB/LibriSpeech_tar/$subset
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

for dest in "test-clean" "test-other"; do
#for dest in "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./bitfit.sh bitfit_"$spk_id"_q_fc1 $spk_id bear
	done
done

#prompt_tuning_100_rand_"$spk_id" $spk_id $subset
