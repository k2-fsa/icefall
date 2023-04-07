dl_dir=/DB/LibriSpeech_tar/vox_v3
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

for dest in "teat-clean" "test-othr"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./test.sh $spk_id
	done
done
