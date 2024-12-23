export PYTHONPATH=$PYTHONPATH:/home/yuekaiz/icefall_matcha

accelerate launch f5-tts/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "seedtts_test_zh" -nfe 16
