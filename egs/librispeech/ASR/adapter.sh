#Dl_dir=/DB/LibriSpeech_tar/vox
dl_dir=/home/work/workspace/LibriSpeech/vox

for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./run_adapter.sh "$spk_id"_adapter $spk_id
	done
done
