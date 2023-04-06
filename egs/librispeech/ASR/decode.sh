dl_dir=/DB/LibriSpeech_tar/vox_v3

for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./test.sh $spk_id
	done
done
