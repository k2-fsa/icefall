# Introduction

A bilingual Japanese-English ASR model developed by the developers of ReazonSpeech that utilizes ReazonSpeech and the English subset of Multilingual LibriSpeech (MLS English), .

**ReazonSpeech** is an open-source dataset that contains a diverse set of natural Japanese speech, collected from terrestrial television streams. It contains more than 35,000 hours of audio.

**Multilingual LibriSpeech (MLS)** is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish. It includes about 44.5K hours of English and a total of about 6K hours for other languages. This icefall training recipe was created for the restructured version of the English split of the dataset available on Hugging Face from `parler-tts` [here](https://huggingface.co/datasets/parler-tts/mls_eng).


# Training Sets

1. ReazonSpeech (Japanese)
2. Multilingual LibriSpeech (English)

|Datset| Number of hours| URL|
|---|---:|---|
|**TOTAL**|79,500|---|
|MLS English|44,500|https://huggingface.co/datasets/parler-tts/mls_eng|
|ReazonSpeech (all)|35,000|https://huggingface.co/datasets/reazon-research/reazonspeech|

# Usage

This recipe relies on the `mls_english` recipe and the `reazonspeech` recipe. 

To be able to use the `multi_ja_en` recipe, you must first run the `prepare.sh` scripts in both the `mls_english` recipe and the `reazonspeech` recipe.

This recipe does not enforce data balance: please ensure that the `mls_english` and `reazonspeech` datasets prepared above are balanced to your liking (you may use the utility script `create_subsets_greedy.py` in the `mls_english` recipe to create a custom-sized MLS English sub-dataset).

Steps for model training:

0. Run `../../mls_english/ASR/prepare.sh` and `../../reazonspeech/ASR/prepare.sh`
1. Run `./prepare.sh`
2. Run `update_cutset_paths.py` (we will soon add this to `./prepare.sh`)
3. Run `zipformer/train.py` (see example arguments inside the file)


