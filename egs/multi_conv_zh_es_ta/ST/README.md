# HENT-SRT

This repository contains a **speech-to-text translation (ST)** recipe accompanying our IWSLT 2025 paper:

**HENT-SRT: Hierarchical Efficient Neural Transducer with Self-Distillation for Joint Speech Recognition and Translation**  
Paper: <https://arxiv.org/abs/2506.02157>

## Datasets

The recipe combines three conversational, 3-way parallel ST corpora:

- **Tunisian–English (IWSLT’22 TA)**  
  Lhotse recipe: <https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/iwslt22_ta.py>

- **Fisher Spanish**  
  Reference: <https://aclanthology.org/2013.iwslt-papers.14>

- **HKUST (Mandarin Telephone Speech)**  
  Reference: <https://arxiv.org/abs/2404.11619>

> **Data access:** Fisher and HKUST require an institutional LDC subscription.  
> **Recipe status:** Lhotse recipes for Fisher Spanish and HKUST are in progress and will be finalized soon.



## Zipformer Multi-joiner ST
This model is similar to https://www.isca-archive.org/interspeech_2023/wang23oa_interspeech.pdf, but 
our system uses zipformer encoder with a pruned transducer and stateless decoder

| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | modified beam search | 41.6      | 16.3       | --epoch 20, --avg 13, beam(20),  |
| hkust      | modified beam search | 23.8      | 10.4       | --epoch 20, --avg 13, beam(20),  |
| fisher\_sp | modified beam search | 18.0      | 31.0       | --epoch 20, --avg 13, beam(20),  |


## Hent-SRT offline

| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | modified beam search | 41.4      | 20.6       | --epoch 20, --avg 13, beam(20),  BP 1 |
| hkust      | modified beam search | 22.8      | 14.7       | --epoch 20, --avg 13, beam(20),  BP 1 |
| fisher\_sp | modified beam search | 17.8      | 33.7       | --epoch 20, --avg 13, beam(20),  BP 1 |

## Hent-SRT streaming

| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | greedy search | 46.2      | 17.3       | --epoch 20, --avg 13, BP 2, chunk-size 64, left-context-frames 128, max-sym-per-frame 20  |
| hkust      | greedy search | 27.3      | 11.2       | --epoch 20, --avg 13, BP 2, chunk-size 64, left-context-frames 128, max-sym-per-frame 20|
| fisher\_sp | greedy search | 22.7      | 30.8     | --epoch 20, --avg 13, BP 2, chunk-size 64, left-context-frames 128, max-sym-per-frame 20 |

See [RESULTS](/egs/multi_conv_zh_es_ta/ST/RESULTS.md) for details.