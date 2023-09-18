# Introduction

This is a weakly supervised ASR recipe for the LibriSpeech (clean 100 hours) dataset. We train a
conformer model using Bypass Temporal Classification (BTC)/Omni-temporal Classification (OTC) with transcripts with synthetic errors. In this README, we will describe
the task and the BTC/OTC training process.

Note that OTC is an extension of BTC and supports all BTC functions. Therefore, in the following, we only describe OTC.
## Task
We propose BTC/OTC to directly train an ASR system leveraging weak supervision, i.e., speech with non-verbatim transcripts. This is achieved by using a special token $\star$ to model uncertainties (i.e., substitution errors, insertion errors, and deletion errors) 
within the WFST framework during training.


<div style="display: flex;flex; justify-content: space-between">
  <figure style="flex: 2; text-align: center; margin: 5px;">
    <img src="figures/sub.png" alt="Image 1" width="25%" />

  </figure>
  <figure style="flex: 2; text-align: center; margin: 5px;">
    <img src="figures/ins.png" alt="Image 2" width="25%" />

  </figure>
  <figure style="flex: 2; text-align: center;margin: 5px;">
    <img src="figures/del.png" alt="Image 3" width="25%" />

  </figure>
</div>
<figcaption> Examples of errors (substitution, insertion, and deletion) in the transcript. The grey box is the verbatim transcript and the red box is the inaccurate transcript. Inaccurate words are marked in bold.</figcaption> <br><br>


We modify $G(\mathbf{y})$ by adding self-loop arcs into each state and bypass arcs into each arc. 
  <p align="center">
    <img src="figures/otc_g.png" alt="Image Alt Text" width="50%" />

  </p>

After composing the modified WFST $G_{\text{otc}}(\mathbf{y})$ with $L$ and $T$, the OTC training graph is shown in this figure:
<figure style="text-align: center">
  <img src="figures/otc_training_graph.drawio.png" alt="Image Alt Text" />
  <figcaption>OTC training graph. The self-loop arcs and bypass arcs are highlighted in green and blue, respectively.</figcaption>
</figure>

The $\star$ is represented as the average probability of all non-blank tokens.
  <p align="center">
    <img src="figures/otc_emission.drawio.png" width="50%" />
  </p>

The weight of $\star$ is the log average probability of "a" and "b": $\log \frac{e^{-1.2} + e^{-2.3}}{2} = -1.6$ and $\log \frac{e^{-1.9} + e^{-0.5}}{2} = -1.0$ for 2 frames.

## Description of the recipe
### Preparation
```
feature_dir="data/ssl"
manifest_dir="${feature_dir}"
lang_dir="data/lang"
lm_dir="data/lm"
exp_dir="conformer_ctc2/exp"
otc_token="<star>"

./prepare.sh \
  --feature-dir "${feature_dir}" \
  --lang-dir "${lang_dir}" \
  --lm-dir "${lm_dir}" \
  --otc-token "${otc_token}" 
```
This script adds the 'otc_token' ('\<star\>') and its corresponding sentence-piece ('▁\<star\>') to 'words.txt' and 'tokens.txt,' respectively. Additionally, it computes SSL features using the 'wav2vec2-base' model. (You can use GPU to accelerate feature extraction).

### Making synthetic errors to the transcript [optional]
```
sub_er=0.17
ins_er=0.17
del_er=0.17
synthetic_train_mainfest="librispeech_cuts_train-clean-100_${sub_er}_${ins_er}_${del_er}.jsonl.gz"

./local/make_error_cutset.py \
  --input-cutset "${feature_dir}/librispeech_cuts_train-clean-100.jsonl.gz" \
  --words-file "${lang_dir}/words.txt" \
  --sub-error-rate "${sub_er}" \
  --ins-error-rate "${ins_er}" \
  --del-error-rate "${del_er}" \
  --output-cutset "${manifest_dir}/${synthetic_train_manifest}"
```
This script generates synthetic substitution, insertion, and deletion errors in the transcript with ratios 'sub_er', 'ins_er', and 'del_er', respectively. The original transcript is saved as 'verbatim transcript' in the cutset, along with information on how the transcript is corrupted:
  - '[hello]' indicates the original word is substituted by 'hello'
  - '[]' indicates an extra word is inserted into the transcript
  - '-hello-' indicates the word 'hello' is deleted from the transcript
So if the original transcript is "have a nice day" and the synthetic one is "a very good day", the 'verbatim transcript' would be:
```
original:  have  a      nice  day
synthetic:       a very good  day
verbatim: -have- a  [] [good] day
```

### Training
```
allow_bypass_arc=true
allow_self_loop_arc=true

initial_bypass_weight=-19
initial_self_loop_weight=3.75

bypass_weight_decay=0.975
self_loop_weight_decay=0.999

show_alignment=true

export CUDA_VISIBLE_DEVICES="0,1,2,3"
./conformer_ctc2/train.py \
  --world-size 4 \
  --manifest-dir "${manifest–dir}" \
  --train-manifest "${train_manifest}" \
  --exp-dir "${exp_dir}" \
  --lang-dir "${lang_dir}" \
  --otc-token "${otc_token}" \
  --allow-bypass-arc "${allow_bypass_arc}" \
  --allow-self-loop-arc "${allow_self_loop_arc}" \
  --initial-bypass-weight "${initial_bypass_weight}" \
  --initial-self-loop-weight "${initial_self_loop_weight}" \
  --bypass-weight-decay "${bypass_weight_decay}" \
  --self-loop-weight-decay "${self_loop_weight_decay}" \
  --show-alignment "${show_alingment}"
```
The bypass arc deals with substitution and insertion errors, while the self-loop arc deals with deletion errors. Using "--show-alignment" would print the best alignment during training, which is very helpful for tuning hyperparameters and debugging.

### Decoding
```
export CUDA_VISIBLE_DEVICES="0"
python conformer_ctc2/decode.py \
  --exp-dir "${exp_dir}" \
  --lang-dir "${lang_dir}" \
  --lm-dir "${lm_dir}" 
```
