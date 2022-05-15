# SPGISpeech

SPGISpeech consists of 5,000 hours of recorded company earnings calls and their respective
transcriptions. The original calls were split into slices ranging from 5 to 15 seconds in
length to allow easy training for speech recognition systems. Calls represent a broad
cross-section of international business English; SPGISpeech contains approximately 50,000
speakers, one of the largest numbers of any speech corpus, and offers a variety of L1 and
L2 English accents. The format of each WAV file is single channel, 16kHz, 16 bit audio.

Transcription text represents the output of several stages of manual post-processing.
As such, the text contains polished English orthography following a detailed style guide,
including proper casing, punctuation, and denormalized non-standard words such as numbers
and acronyms, making SPGISpeech suited for training fully formatted end-to-end models.

Official reference:

O’Neill, P.K., Lavrukhin, V., Majumdar, S., Noroozi, V., Zhang, Y., Kuchaiev, O., Balam,
J., Dovzhenko, Y., Freyberg, K., Shulman, M.D., Ginsburg, B., Watanabe, S., & Kucsko, G.
(2021). SPGISpeech: 5, 000 hours of transcribed financial audio for fully formatted
end-to-end speech recognition. ArXiv, abs/2104.02014.

ArXiv link: https://arxiv.org/abs/2104.02014

## Performance Record

| Decoding method           | val WER    | val CER |
|---------------------------|------------|---------|
| greedy search             | 2.40       |  0.99   |
| modified beam search      | 2.24       |  0.91   |
| fast beam search          | 2.35       |  0.97   |

See [RESULTS](/egs/spgispeech/ASR/RESULTS.md) for details.
