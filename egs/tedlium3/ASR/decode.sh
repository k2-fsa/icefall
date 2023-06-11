for i in {0..10}; do
	spk_id=${spk#*$dest\/}
	echo $spk_id
	./test.sh $spk_id lora_rank6_$spk_id
done
