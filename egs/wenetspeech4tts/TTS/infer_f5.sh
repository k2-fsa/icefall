export PYTHONPATH=$PYTHONPATH:/home/yuekaiz/icefall_matcha
#bigvganinference
model_path=/home/yuekaiz/HF/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt
manifest=/home/yuekaiz/HF/valle_wenetspeech4tts_demo/wenetspeech4tts.txt
manifest=/home/yuekaiz/seed_tts_eval/seedtts_testset/zh/meta_head.lst
# get wenetspeech4tts
manifest_base_stem=$(basename $manifest)
mainfest_base_stem=${manifest_base_stem%.*}
output_dir=./results/f5-tts-pretrained/$mainfest_base_stem


pip install sherpa-onnx bigvganinference lhotse kaldialign sentencepiece
accelerate launch f5-tts/infer.py --nfe 16 --model-path $model_path --manifest-file $manifest --output-dir $output_dir || exit 1

bash local/compute_wer.sh $output_dir $manifest
