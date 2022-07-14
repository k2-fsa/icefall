# MGB2

The Multi-Dialect Broadcast News Arabic Speech Recognition (MGB-2):
The second edition of the Multi-Genre Broadcast (MGB-2) Challenge is
an evaluation of speech recognition and lightly supervised alignment
using TV recordings in Arabic. The speech data is broad and multi-genre,
spanning the whole range of TV output, and represents a challenging task for
speech technology. In 2016, the challenge featured two new Arabic tracks based
on TV data from Aljazeera. It was an official challenge at the 2016 IEEE
Workshop on Spoken Language Technology. The 1,200 hours MGB-2: from Aljazeera
TV programs have been manually captioned with no timing information.
QCRI Arabic ASR system has been used to recognize all programs. The ASR output
was used to align the manual captioning and produce speech segments for
training speech recognition. More than 20 hours from 2015 programs have been
transcribed verbatim and manually segmented. This data is split into a
development set of 10 hours, and a similar evaluation set of 10 hours.
Both the development and evaluation data have been released in the 2016 MGB
challenge

Official reference:

Ali, Ahmed, et al. "The MGB-2 challenge: Arabic multi-dialect broadcast media recognition." 
2016 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2016.

IEEE link: https://ieeexplore.ieee.org/abstract/document/7846277

## Performance Record (after 3 epochs)

| Decoding method           | dev WER    | test WER |
|---------------------------|------------|---------|
| attention-decoder         | 27.87      |  26.12   |
| whole-lattice-rescoring   | 25.32      |  23.53   |


See [RESULTS](/egs/mgb2/ASR/RESULTS.md) for details.
