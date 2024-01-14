# GigaSpeech
GigaSpeech, an evolving, multi-domain English
speech recognition corpus with 10,000 hours of high quality labeled
audio, collected from audiobooks, podcasts
and YouTube, covering both read and spontaneous speaking styles,
and a variety of topics, such as arts, science, sports, etc. More details can be found: https://github.com/SpeechColab/GigaSpeech

## Download

Apply for the download credentials and download the dataset by following https://github.com/SpeechColab/GigaSpeech#download. Then create a symlink
```bash
ln -sfv /path/to/GigaSpeech download/GigaSpeech
```

## Performance Record
|                                |  Dev  | Test  |
|--------------------------------|-------|-------|
|           `zipformer`          | 10.25 | 10.38 |
|         `conformer_ctc`        | 10.47 | 10.58 |
| `pruned_transducer_stateless2` | 10.40 | 10.51 |

See [RESULTS](/egs/gigaspeech/ASR/RESULTS.md) for details.
