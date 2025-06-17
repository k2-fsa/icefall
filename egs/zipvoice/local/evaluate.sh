export CUDA_VISIBLE_DEVICES="0"
export PYTHONWARNINGS=ignore
export PYTHONPATH=../../:$PYTHONPATH

# Uncomment this if you have trouble connecting to HuggingFace
# export HF_ENDPOINT=https://hf-mirror.com

start_stage=1
end_stage=3

# Models used for SIM-o evaluation.
# SV model wavlm_large_finetune.pth is downloaded from https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
# SSL model wavlm_large.pt is downloaded from https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt
sv_model_path=model/UniSpeech/wavlm_large_finetune.pth
wavlm_model_path=model/s3prl/wavlm_large.pt

# Models used for UTMOS evaluation.
# wget https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt -P model/huggingface/utmos/utmos.pt
# wget https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/wav2vec_small.pt -P model/huggingface/utmos/wav2vec_small.pt
utmos_model_path=model/huggingface/utmos/utmos.pt
wav2vec_model_path=model/huggingface/utmos/wav2vec_small.pt


if [ $start_stage -le 1 ] && [ $end_stage -ge 1 ]; then 

        echo "=====Evaluate for Seed-TTS test-en======="
        test_list=testset/test_seedtts_en.tsv
        wav_path=results/zipvoice_seedtts_en

        echo $wav_path
        echo "-----Computing SIM-o-----"
        python3 local/evaluate_sim.py \
                --sv-model-path ${sv_model_path} \
                --ssl-model-path ${wavlm_model_path} \
                --eval-path ${wav_path} \
                --test-list ${test_list}

        echo "-----Computing WER-----"
        python3 local/evaluate_wer_seedtts.py \
                --test-list ${test_list} \
                --wav-path ${wav_path} \
                --lang "en" 

        echo "-----Computing UTSMOS-----"
        python3 local/evaluate_utmos.py \
                --wav-path ${wav_path} \
                --utmos-model-path ${utmos_model_path} \
                --ssl-model-path ${wav2vec_model_path}

fi

if [ $start_stage -le 2 ] && [ $end_stage -ge 2 ]; then 
        echo "=====Evaluate for Seed-TTS test-zh======="
        test_list=testset/test_seedtts_zh.tsv
        wav_path=results/zipvoice_seedtts_zh

        echo $wav_path
        echo "-----Computing SIM-o-----"
        python3 local/evaluate_sim.py \
                --sv-model-path ${sv_model_path} \
                --ssl-model-path ${wavlm_model_path} \
                --eval-path ${wav_path} \
                --test-list ${test_list}

        echo "-----Computing WER-----"
        python3 local/evaluate_wer_seedtts.py \
                --test-list ${test_list} \
                --wav-path ${wav_path} \
                --lang "zh" 

        echo "-----Computing UTSMOS-----"
        python3 local/evaluate_utmos.py \
                --wav-path ${wav_path} \
                --utmos-model-path ${utmos_model_path} \
                --ssl-model-path ${wav2vec_model_path}
fi

if [ $start_stage -le 3 ] && [ $end_stage -ge 3 ]; then 
        echo "=====Evaluate for Librispeech test-clean======="
        test_list=testset/test_librispeech_pc_test_clean.tsv
        wav_path=results/zipvoice_librispeech_test_clean

        echo $wav_path
        echo "-----Computing SIM-o-----"
        python3 local/evaluate_sim.py \
                --sv-model-path ${sv_model_path} \
                --ssl-model-path ${wavlm_model_path} \
                --eval-path ${wav_path} \
                --test-list ${test_list}

        echo "-----Computing WER-----"
        python3 local/evaluate_wer_hubert.py \
                --test-list ${test_list} \
                --wav-path ${wav_path} \

        echo "-----Computing UTSMOS-----"
        python3 local/evaluate_utmos.py \
                --wav-path ${wav_path} \
                --utmos-model-path ${utmos_model_path} \
                --ssl-model-path ${wav2vec_model_path}

fi