num_epochs=30
for ((i=$num_epochs; i>=1; i--));
do
    for ((j=1; j<=$i; j++));
    do
        python3 ./pruned_transducer_stateless7_streaming/decode.py \
            --exp-dir exp \
            --lang data/lang_char \
            --epoch $i \
            --avg $j \
            --max-duration 180 \
            --decoding-method modified_beam_search
    done
done
