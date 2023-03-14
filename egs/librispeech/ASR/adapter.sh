dl_dir=/DB/LibriSpeech_tar/vox

for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		if [$spk_id -ne 1089]; then
			./run_adapter.sh "$spk_id"_adapter $spk_id
		fi
	done
done
