if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/tmp/icefall"
else
    export PYTHONPATH="${PYTHONPATH}:/tmp/icefall"
fi

CUDA_VISIBLE_DEVICES=3 python ./conformer_ctc/decode.py \
    --method ctc-decoding \
    --max-duration 10 \
    --epoch 77 \
    --avg 10
