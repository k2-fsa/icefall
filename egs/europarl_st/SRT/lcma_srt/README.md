# LCMA-SRT: Language-Conditional Mixture-of-Experts Adapters for Joint Multilingual Speech Recognition and Translation

This recipe implements LCMA-SRT on the [Europarl-ST](https://www.mllp.upv.es/europarl-st/) dataset (9 languages, 72 translation directions).

Paper: [ACL 2026](https://aclanthology.org/2026.acl-long.1634/)

## Model Architecture

LCMA-SRT augments a hierarchical transducer (Zipformer encoder) with:
- **SRC-MoE adapter**: Source-conditioned Mixture-of-Experts adapter after the ASR encoder to reduce cross-language interference
- **TGT-MoE adapter**: Target-conditioned Mixture-of-Experts adapter for the ST encoder to stabilize target-language generation

## Performance

| Model | WER (%) ↓ | Avg BLEU ↑ | Avg COMET ↑ | LMR (%) ↓ |
|-------|-----------|-----------|-------------|-----------|
| HENT-SRT-M2O×9 | 23.28 | 15.3 | 0.575 | 0.65 |
| HENT-SRT-M2M | 16.65 | 4.3 | 0.436 | 84.95 |
| **LCMA-SRT** | **15.71** | **20.5** | **0.651** | **0.75** |

## Usage

### Data Preparation

```bash
cd egs/europarl_st/SRT
bash prepare.sh --stage 0 --stop-stage 5
```

See [local/README.md](local/README.md) for detailed documentation of each preprocessing step.

### Training

#### Stage 1: Multilingual ASR Pretraining (with SRC-MoE)

```bash
bash lcma_srt/cr_ctc_sc_moe.sh
```

#### Stage 2: Many-to-Many Joint ASR+ST Training

```bash
bash lcma_srt/lcma_srt.sh
```

### Decoding

#### Stage 1: ASR Decoding

```bash
bash lcma_srt/decode_cr_ctc_sc_moe.sh
```

#### Stage 2: Joint ASR+ST Decoding

```bash
bash lcma_srt/decode_lcma_srt.sh
```

## Pre-trained Models

Pre-trained checkpoints are available on [OSF](https://osf.io/rnuhv/).

## Citation

```bibtex
@inproceedings{li2026lcma,
  title={LCMA-SRT: Language-Conditional Mixture-of-Experts Adapters for Joint Multilingual Speech Recognition and Translation},
  author={Li, Nanjie and Guo, Xiaoyong and Huang, Hao and Haihua, Xu and Shi, Wei},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={35363--35377},
  year={2026}
}
```
