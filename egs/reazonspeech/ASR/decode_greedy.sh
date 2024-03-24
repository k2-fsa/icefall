num_epochs=40
for ((i=$num_epochs; i>=1; i--));
do
    for ((j=1; j<=$i; j++));
    do
        python3 ./zipformer/decode.py \
            --epoch $i \
            --avg $j \
            --exp-dir zipformer/exp-large \
            --max-duration 600 \
            --causal 0 \
            --decoding-method greedy_search \
            --num-encoder-layers 2,2,4,5,4,2 \
            --feedforward-dim 512,768,1536,2048,1536,768 \
            --encoder-dim 192,256,512,768,512,256 \
            --encoder-unmasked-dim 192,192,256,320,256,192 \
            --lang data/lang_char \
            --blank-penalty 0
    done
done
