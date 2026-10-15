# Introduction

This recipe includes ASR models (zipformer, zipformer-hat, zipformer-hat-lid) trained and evaluated on SEAME dataset.
The SEAME corpora is Singaporean Codeswitched English and Mandarin.

This corpus comes defined with a training split and two development splits:

train -- A mix of codeswitched, Mandarin and Singaporean English
dev_sge -- A set of primarily Singaporean English though there is codeswitching  
dev_man -- A set of primarily Mandarin though there is also some codeswitching


[./RESULTS.md](./RESULTS.md) contains the latest results.

# Zipformer-hat

Zipformer with hybrid autoregressive transducer (HAT) loss https://arxiv.org/abs/2003.07705
see https://github.com/k2-fsa/icefall/pull/1291

# Zipformer-hat-lid

Zipformer-hat with auxiliary LID encoder and blank sharing for synchronization between ASR and LID as described here (will be shared soon)

