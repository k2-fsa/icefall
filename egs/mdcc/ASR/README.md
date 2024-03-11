# Introduction

Multi-Domain Cantonese Corpus (MDCC), consists of 73.6 hours of clean read speech paired with 
transcripts, collected from Cantonese audiobooks from Hong Kong. It comprises philosophy, 
politics, education, culture, lifestyle and family domains, covering a wide range of topics. 

Manuscript can be found at: https://arxiv.org/abs/2201.02419

# Transducers



|                                       | Encoder             | Decoder            | Comment                     |
|---------------------------------------|---------------------|--------------------|-----------------------------|
| `zipformer`                           | Upgraded Zipformer | Embedding + Conv1d | The latest recipe with context-size set to 1 |

The decoder is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
