cd data/

log "Preparing LM data..."
mkdir -p lm_training_data
mkdir -p lm_dev_data
mkdir -p lm_test_data

log "aidatatang_200zh"
gunzip -c manifests/aidatatang_200zh/aidatatang_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/aidatatang_train_text

gunzip -c manifests/aidatatang_200zh/aidatatang_supervisions_dev.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/aidatatang_dev_text

gunzip -c manifests/aidatatang_200zh/aidatatang_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/aidatatang_test_text

log "aishell"
gunzip -c manifests/aishell/aishell_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/aishell_train_text

gunzip -c manifests/aishell/aishell_supervisions_dev.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/aishell_dev_text

gunzip -c manifests/aishell/aishell_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/aishell_test_text

log "aishell2"
gunzip -c manifests/aishell2/aishell2_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/aishell2_train_text

gunzip -c manifests/aishell2/aishell2_supervisions_dev.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/aishell2_dev_text

gunzip -c manifests/aishell2/aishell2_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/aishell2_test_text

log "aishell4"
gunzip -c manifests/aishell4/aishell4_supervisions_train_L.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/aishell4_train_L_text

gunzip -c manifests/aishell4/aishell4_supervisions_train_M.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/aishell4_train_M_text

gunzip -c manifests/aishell4/aishell4_supervisions_train_S.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/aishell4_train_S_text

gunzip -c manifests/aishell4/aishell4_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/aishell4_test_text

log "alimeeting"
gunzip -c manifests/alimeeting/alimeeting-far_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/alimeeting-far_train_text

gunzip -c manifests/alimeeting/alimeeting-far_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/alimeeting-far_test_text

gunzip -c manifests/alimeeting/alimeeting-far_supervisions_eval.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/alimeeting-far_eval_text

log "kespeech"
gunzip -c manifests/kespeech/kespeech-asr_supervisions_dev_phase1.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/kespeech_dev_phase1_text

gunzip -c manifests/kespeech/kespeech-asr_supervisions_dev_phase2.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/kespeech_dev_phase2_text

gunzip -c manifests/kespeech/kespeech-asr_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/kespeech_test_text

gunzip -c manifests/kespeech/kespeech-asr_supervisions_train_phase1.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/kespeech_train_phase1_text

gunzip -c manifests/kespeech/kespeech-asr_supervisions_train_phase2.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/kespeech_train_phase2_text

log "magicdata"
gunzip -c manifests/magicdata/magicdata_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/magicdata_train_text

gunzip -c manifests/magicdata/magicdata_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/magicdata_test_text

gunzip -c manifests/magicdata/magicdata_supervisions_dev.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/magicdata_dev_text

log "stcmds"
gunzip -c manifests/stcmds/stcmds_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/stcmds_train_text

log "primewords"
gunzip -c manifests/primewords/primewords_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/primewords_train_text

log "thchs30"
gunzip -c manifests/thchs30/thchs_30_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/thchs30_train_text

gunzip -c manifests/thchs30/thchs_30_supervisions_test.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/thchs30_test_text

gunzip -c manifests/thchs30/thchs_30_supervisions_dev.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/thchs30_dev_text

log "wenetspeech"
gunzip -c manifests/wenetspeech/wenetspeech_supervisions_L.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_training_data/wenetspeech_L_text

gunzip -c manifests/wenetspeech/wenetspeech_supervisions_DEV.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_dev_data/wenetspeech_DEV_text

gunzip -c manifests/wenetspeech/wenetspeech_supervisions_TEST_MEETING.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/wenetspeech_TEST_MEETING_text

gunzip -c manifests/wenetspeech/wenetspeech_supervisions_TEST_NET.jsonl.gz \
    | jq '.text' \
    | sed 's/"//g' \
    | ../local/tokenize_for_lm_training.py -t "char" \
    > lm_test_data/wenetspeech_TEST_NET_text

for f in aidatatang_train_text aishell2_train_text aishell4_train_L_text aishell4_train_M_text aishell4_train_S_text aishell_train_text alimeeting-far_train_text kespeech_train_phase1_text kespeech_train_phase2_text magicdata_train_text primewords_train_text stcmds_train_text thchs30_train_text wenetspeech_L_text; do
  cat lm_training_data/$f >> lm_training_data/lm_training_text
done

for f in aidatatang_test_text aishell4_test_text alimeeting-far_test_text  thchs30_test_text wenetspeech_TEST_NET_text aishell2_test_text aishell_test_text kespeech_test_text magicdata_test_text wenetspeech_TEST_MEETING_text; do
  cat lm_test_data/$f >> lm_test_data/lm_test_text
done

for f in aidatatang_dev_text aishell_dev_text kespeech_dev_phase1_text thchs30_dev_text aishell2_dev_text alimeeting-far_eval_text kespeech_dev_phase2_text magicdata_dev_text wenetspeech_DEV_text; do
  cat lm_dev_data/$f >> lm_dev_data/lm_dev_text
done

cd ../
