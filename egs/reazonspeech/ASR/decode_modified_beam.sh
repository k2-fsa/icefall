num_epochs=30
for ((i=$num_epochs; i>=20; i--));
do
    for avg in 12 11 10 9 8 7 6 5;
    do
        python3 ./zipformer/decode.py \
            --epoch $i \
            --avg $avg \
            --exp-dir zipformer/exp \
            --max-duration 450 \
            --lang data/lang_char \
            --decoding-method modified_beam_search \
	    --blank-penalty 2.5
    done
done
