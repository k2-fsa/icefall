Note: This recipe is trained with the codes from this PR https://github.com/k2-fsa/icefall/pull/375
# Pre-trained Transducer-Stateless2 models for the Aidatatang_200zh dataset with icefall.
The model was trained on full [Aidatatang_200zh](https://www.openslr.org/62) with the scripts in [icefall](https://github.com/k2-fsa/icefall) based on the latest version k2.
## Training procedure
The main repositories are list below, we will update the training and decoding scripts with the update of version.
k2: https://github.com/k2-fsa/k2
icefall: https://github.com/k2-fsa/icefall
lhotse: https://github.com/lhotse-speech/lhotse
* Install k2 and lhotse, k2 installation guide refers to https://k2.readthedocs.io/en/latest/installation/index.html, lhotse refers to https://lhotse.readthedocs.io/en/latest/getting-started.html#installation. I think the latest version would be ok. And please also install the requirements listed in icefall.
* Clone icefall(https://github.com/k2-fsa/icefall) and check to the commit showed above.
```
git clone https://github.com/k2-fsa/icefall
cd icefall
```
* Preparing data.
```
cd egs/aidatatang_200zh/ASR
bash ./prepare.sh
```
* Training
```
export CUDA_VISIBLE_DEVICES="0,1"
./pruned_transducer_stateless2/train.py \
                  --world-size 2 \
                  --num-epochs 30 \
                  --start-epoch 0 \
                  --exp-dir pruned_transducer_stateless2/exp \
                  --lang-dir data/lang_char \
                  --max-duration 250
```
## Evaluation results
The decoding results (WER%) on Aidatatang_200zh(dev and test) are listed below, we got this result by averaging models from epoch 11 to 29.
The WERs are
|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 5.53       | 6.59       | --epoch 29, --avg 19, --max-duration 100 |
| modified beam search (beam size 4) | 5.27       | 6.33       | --epoch 29, --avg 19, --max-duration 100 |
| fast beam search (set as default)  | 5.30       | 6.34       | --epoch 29, --avg 19, --max-duration 1500|
