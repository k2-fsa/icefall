num_epochs=30
for ((i=$num_epochs; i>=15; i--));
do
    for ((j=1; j<=$i; j++));
    do
        python3 ./zipformer/decode.py \
            --epoch $i \
            --avg $j \
            --exp-dir zipformer/exp \
            --max-duration 450 \
            --lang data/lang_char \
            --decoding-method greedy_search \
	    --blank-penalty 2
    done
done
