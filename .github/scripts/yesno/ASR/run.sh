#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/yesno/ASR

log "data preparation"
./prepare.sh

log "training"
python3 ./tdnn/train.py

log "decoding"
python3 ./tdnn/decode.py

log "export to pretrained.pt"

python3 ./tdnn/export.py --epoch 14 --avg 2

python3 ./tdnn/pretrained.py \
  --checkpoint ./tdnn/exp/pretrained.pt \
  --HLG ./data/lang_phone/HLG.pt \
  --words-file ./data/lang_phone/words.txt \
  download/waves_yesno/0_0_0_1_0_0_0_1.wav \
  download/waves_yesno/0_0_1_0_0_0_1_0.wav

log "Test exporting to torchscript"
python3 ./tdnn/export.py --epoch 14 --avg 2 --jit 1

python3 ./tdnn/jit_pretrained.py \
  --nn-model ./tdnn/exp/cpu_jit.pt \
  --HLG ./data/lang_phone/HLG.pt \
  --words-file ./data/lang_phone/words.txt \
  download/waves_yesno/0_0_0_1_0_0_0_1.wav \
  download/waves_yesno/0_0_1_0_0_0_1_0.wav

log "Test exporting to onnx"
python3 ./tdnn/export_onnx.py --epoch 14 --avg 2

log "Test float32 model"
python3 ./tdnn/onnx_pretrained.py \
  --nn-model ./tdnn/exp/model-epoch-14-avg-2.onnx \
  --HLG ./data/lang_phone/HLG.pt \
  --words-file ./data/lang_phone/words.txt \
  download/waves_yesno/0_0_0_1_0_0_0_1.wav \
  download/waves_yesno/0_0_1_0_0_0_1_0.wav

log "Test int8 model"
python3 ./tdnn/onnx_pretrained.py \
  --nn-model ./tdnn/exp/model-epoch-14-avg-2.int8.onnx \
  --HLG ./data/lang_phone/HLG.pt \
  --words-file ./data/lang_phone/words.txt \
  download/waves_yesno/0_0_0_1_0_0_0_1.wav \
  download/waves_yesno/0_0_1_0_0_0_1_0.wav

log "Test decoding with H"
python3 ./tdnn/export.py --epoch 14 --avg 2 --jit 1

python3 ./tdnn/jit_pretrained_decode_with_H.py \
    --nn-model ./tdnn/exp/cpu_jit.pt \
    --H ./data/lang_phone/H.fst \
    --tokens ./data/lang_phone/tokens.txt \
    ./download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    ./download/waves_yesno/0_0_1_0_0_0_1_0.wav \
    ./download/waves_yesno/0_0_1_0_0_1_1_1.wav

log "Test decoding with HL"
python3 ./tdnn/export.py --epoch 14 --avg 2 --jit 1

python3 ./tdnn/jit_pretrained_decode_with_HL.py \
    --nn-model ./tdnn/exp/cpu_jit.pt \
    --HL ./data/lang_phone/HL.fst \
    --words ./data/lang_phone/words.txt \
    ./download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    ./download/waves_yesno/0_0_1_0_0_0_1_0.wav \
    ./download/waves_yesno/0_0_1_0_0_1_1_1.wav

log "Show generated files"
ls -lh tdnn/exp
ls -lh data/lang_phone
