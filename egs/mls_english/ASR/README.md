# Introduction



**Multilingual LibriSpeech (MLS)** is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish. It includes about 44.5K hours of English and a total of about 6K hours for other languages. This icefall training recipe was created for the restructured version of the English split of the dataset available on Hugging Face below.


The dataset is available on Hugging Face. For more details, please visit:

- Dataset: https://huggingface.co/datasets/parler-tts/mls_eng
- Original MLS dataset link: https://www.openslr.org/94


## On-the-fly feature computation

This recipe currently only supports on-the-fly feature bank computation, since `lhotse` manifests and feature banks are not pre-calculated in this recipe. This should mean that the dataset can be streamed from Hugging Face, but we have not tested this yet. We may add a version that supports pre-calculating features to better match existing recipes.\
<br>

[./RESULTS.md](./RESULTS.md) contains the latest results. This MLS English recipe was primarily developed for use in the ```multi_ja_en``` Japanese-English bilingual pipeline, which is based on MLS English and ReazonSpeech.
