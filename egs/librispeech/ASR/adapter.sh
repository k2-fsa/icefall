dl_dir=/DB/LibriSpeech_tar/vox

for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./pseudo.sh $spk_id
	done
done
