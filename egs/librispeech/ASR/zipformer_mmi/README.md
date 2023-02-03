This recipe implements Zipformer-MMI model.

See https://k2-fsa.github.io/icefall/recipes/Non-streaming-ASR/librispeech/zipformer_mmi.html for detailed tutorials.

It uses **CTC loss for warm-up** and then switches to MMI loss during training.

For decoding, it uses HP (H is ctc_topo, P is token-level bi-gram) as decoding graph. Supported decoding methods are:
- **1best**. Extract the best path from the decoding lattice as the decoding result.
- **nbest**. Extract n paths from the decoding lattice; the path with the highest score is the decoding result.
- **nbest-rescoring-LG**. Extract n paths from the decoding lattice, rescore them with an word-level 3-gram LM, the path with the highest score is the decoding result.
- **nbest-rescoring-3-gram**. Extract n paths from the decoding lattice, rescore them with an token-level 3-gram LM, the path with the highest score is the decoding result.
- **nbest-rescoring-4-gram**. Extract n paths from the decoding lattice, rescore them with an token-level 4-gram LM, the path with the highest score is the decoding result.

Experimental results training on train-clean-100 (epoch-30-avg-10):
- 1best. 6.43 & 17.44
- nbest, nbest-scale=1.2, 6.43 & 17.45
- nbest-rescoring-LG, nbest-scale=1.2, 5.87 & 16.35
- nbest-rescoring-3-gram,  nbest-scale=1.2, 6.19 & 16.57
- nbest-rescoring-4-gram,  nbest-scale=1.2, 5.87 & 16.07

Experimental results training on full librispeech (epoch-30-avg-10):
- 1best. 2.54 & 5.65
- nbest, nbest-scale=1.2, 2.54 & 5.66
- nbest-rescoring-LG, nbest-scale=1.2, 2.49 & 5.42
- nbest-rescoring-3-gram,  nbest-scale=1.2, 2.52 & 5.62
- nbest-rescoring-4-gram,  nbest-scale=1.2, 2.5 & 5.51
