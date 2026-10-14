#!/usr/bin/env bash
set -e
export DISABLE_VERSION_CHECK=1

echo "=== Decoding script started on $(hostname) at $(date) ==="
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export TRANSFORMERS_NO_GIT=1
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

ASR_BPE_MODEL="data/Europarl-ST/bpe/asr9/bpe.model"
AST_BPE_MODEL="data/Europarl-ST/bpe/ast9/bpe.model"

steps=0
epochs=0
avg=1
MODEL_NAME="best-valid-loss.pt"
EXP_DIR="exp/europarl"

TEST_CUTS_PATHS=(
  data/Europarl-ST/manifests/es_de/europarl.es_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_de/europarl.fr_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_de/europarl.it_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_de/europarl.nl_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_de/europarl.pl_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_de/europarl.pt_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_de/europarl.ro_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_de/europarl.en_de.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_en/europarl.de_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_en/europarl.fr_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_en/europarl.it_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_en/europarl.nl_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_en/europarl.pl_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_en/europarl.pt_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_en/europarl.ro_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_en/europarl.es_en.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_es/europarl.de_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_es/europarl.fr_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_es/europarl.it_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_es/europarl.nl_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_es/europarl.pl_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_es/europarl.pt_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_es/europarl.ro_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_es/europarl.en_es.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_fr/europarl.de_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_fr/europarl.es_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_fr/europarl.it_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_fr/europarl.nl_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_fr/europarl.pl_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_fr/europarl.pt_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_fr/europarl.ro_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_fr/europarl.en_fr.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_it/europarl.de_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_it/europarl.fr_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_it/europarl.es_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_it/europarl.nl_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_it/europarl.pl_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_it/europarl.pt_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_it/europarl.ro_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_it/europarl.en_it.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_nl/europarl.de_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_nl/europarl.fr_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_nl/europarl.it_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_nl/europarl.es_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_nl/europarl.pl_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_nl/europarl.pt_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_nl/europarl.ro_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_nl/europarl.en_nl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_pl/europarl.de_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_pl/europarl.fr_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_pl/europarl.it_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_pl/europarl.nl_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_pl/europarl.es_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_pl/europarl.pt_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_pl/europarl.ro_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_pl/europarl.en_pl.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_pt/europarl.de_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_pt/europarl.fr_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_pt/europarl.it_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_pt/europarl.nl_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_pt/europarl.pl_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_pt/europarl.es_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/ro_pt/europarl.ro_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_pt/europarl.en_pt.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/de_ro/europarl.de_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/fr_ro/europarl.fr_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/it_ro/europarl.it_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/nl_ro/europarl.nl_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pl_ro/europarl.pl_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/pt_ro/europarl.pt_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/es_ro/europarl.es_ro.test_cuts.jsonl.gz
  data/Europarl-ST/manifests/en_ro/europarl.en_ro.test_cuts.jsonl.gz
)

compute_cer=False


manifest_dir="data/Europarl-ST/manifests"
decoding_method="modified_beam_search"
beam_size=20


ASR_NUM_LAYERS="2,2,2,2,2"
ASR_FF_DIM="512,768,1024,1024,1024"
ASR_ENC_DIM="192,256,384,512,384"
ASR_UNMASK_DIM="192,192,256,256,256"
downsampling_factor_asr="1,2,4,8,4"
cnn_module_kernel_asr="1,31,15,15,15"
num_heads_asr="4,4,4,8,8"

ST_NUM_LAYERS="2,2,2,2,2"
ST_FF_DIM="512,512,256,256,256"
ST_ENC_DIM="384,512,256,256,256"
ST_UNMASK_DIM="256,256,256,256,192"
downsampling_factor_st="1,2,4,4,4"
cnn_module_kernel_st="15,31,31,15,15"
num_heads_st="8,8,8,8,8"

for TEST_CUTS_PATH in "${TEST_CUTS_PATHS[@]}"; do
  lang_pair_dir=$(basename "$(dirname "$TEST_CUTS_PATH")")
  src_lang=${lang_pair_dir%%_*}
  tgt_lang=${lang_pair_dir##*_}
  if [[ "$src_lang" == "$lang_pair_dir" ]]; then
    tgt_lang=$src_lang
  fi

  decoding_method_dir="modified_beam_search_beam20_${src_lang}_to_${tgt_lang}_cuts_test"
  compute_cer_current=$compute_cer
  tgt_lang_token="<2${tgt_lang}>"
  python lcma_srt/decode.py \
    --iter $steps \
    --avg $avg \
    --use-averaged-model 0 \
    --exp-dir $EXP_DIR \
    --bpe-model-asr ${ASR_BPE_MODEL} \
    --bpe-model-st  ${AST_BPE_MODEL} \
    --manifest-dir $manifest_dir \
    --decoding-method $decoding_method \
    --beam-size $beam_size \
    --max-duration 500 \
    --compute-cer $compute_cer_current \
    --remove-punctuation True \
    --causal 0 \
    --num-encoder-layers-asr ${ASR_NUM_LAYERS} \
    --feedforward-dim-asr ${ASR_FF_DIM} \
    --encoder-dim-asr ${ASR_ENC_DIM} \
    --encoder-unmasked-dim-asr ${ASR_UNMASK_DIM} \
    --num-encoder-layers-st ${ST_NUM_LAYERS} \
    --feedforward-dim-st ${ST_FF_DIM} \
    --encoder-dim-st ${ST_ENC_DIM} \
    --encoder-unmasked-dim-st ${ST_UNMASK_DIM} \
    --downsampling-factor-st ${downsampling_factor_st} \
    --cnn-module-kernel-st ${cnn_module_kernel_st} \
    --num-heads-st ${num_heads_st} \
    --downsampling-factor-asr ${downsampling_factor_asr} \
    --chunk-size -1 \
    --test-name $TEST_CUTS_PATH \
    --left-context-frames -1 \
    --use-ctc-asr 1 \
    --asr-decode 1 \
    --ast-decode 1 \
    --use-ctc-st 1 \
    --blank-penalty-st 2.0 \
    --decoding-method-dir $decoding_method_dir \
    --num-heads-asr ${num_heads_asr} \
    --cnn-module-kernel-asr ${cnn_module_kernel_asr} \
    --output-downsampling-factor-st 1 \
    --decoder-dim-asr 256 \
    --decoder-dim-st 256 \
    --joiner-dim-asr 256 \
    --joiner-dim-st 256 \
    --use-tgt 1 \
    --lang-tgt "${tgt_lang_token}" \
    --force-first-lang 1 \
    --asr-moe 0 \
    --asr-src 0 \
    --ast-moe 0 \
    --ast-tgt 0 \
    --num-experts-asr 0 \
    --num-experts-ast 0 \
    --entropy-reg-asr 0.0 \
    --entropy-reg-ast 0.0 \
    --tgt-langs "en,de,es,fr,it,nl,pl,pt,ro" \
    --srt-langs "en,de,es,fr,it,nl,pl,pt,ro" \
    --dump-moe-routing-stats 0 \
    --enable-st 1 \
    --model-name $MODEL_NAME
done

echo "--- Decoding script finished at $(date) ---"
