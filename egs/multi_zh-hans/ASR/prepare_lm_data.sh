for subset in train dev test; do
    gunzip -c aidatatang_200zh/aidatatang_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > aidatatang_${subset}_text
done

for subset in train dev test; do
    gunzip -c aishell/aishell_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > aishell_${subset}_text
done

for subset in train dev test; do
    gunzip -c aishell2/aishell2_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > aishell2_${subset}_text
done

for subset in train_L train_M train_S test; do
    gunzip -c aishell4/aishell4_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > aishell4_${subset}_text
done

for subset in train test eval; do
    gunzip -c alimeeting/alimeeting-far_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > alimeeting-far_${subset}_text
done

for subset in dev_phase1 dev_phase2 test train_phase1 train_phase2; do
    gunzip -c kespeech/kespeech-asr_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > kespeech_${subset}_text
done

for subset in train test dev; do
    gunzip -c magicdata/magicdata_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > magicdata_${subset}_text
done

for subset in train ; do
    gunzip -c stcmds/stcmds_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > stcmds_${subset}_text
done

for subset in train ; do
    gunzip -c primewords/primewords_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > primewords_${subset}_text
done

for subset in train test dev ; do
    gunzip -c thchs30/thchs_30_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > thchs30_${subset}_text
done

for subset in L DEV TEST_MEETING TEST_NET ; do
    gunzip -c wenetspeech/wenetspeech_supervisions_${subset}.jsonl.gz \
        | jq '.text' \
        | sed 's/"//g' \
        | ../../local/tokenize_for_lm_training.py -t "char" \
        > wenetspeech_${subset}_text
done

cat aidatatang_train_text aishell2_train_text aishell4_train_L_text \
    aishell4_train_M_text aishell4_train_S_text aishell_train_text \
    alimeeting-far_train_text kespeech_train_phase1_text kespeech_train_phase2_text \
    magicdata_train_text primewords_train_text stcmds_train_text \
    thchs30_train_text wenetspeech_L_text > lm_training_text
