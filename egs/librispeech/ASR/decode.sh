dl_dir=/DB/LibriSpeech_tar/vox

for dest in "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./test.sh $spk_id
	done
done
