#dl_dir=/DB/LibriSpeech_tar/vox_10m
dl_dir=/home/work/workspace/LibriSpeech/vox_v3
for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./test.sh $spk_id prompt_tuning_"$spk_id"
	done
done
