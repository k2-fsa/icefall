# Europarl-ST Preprocessing Scripts

A complete preprocessing pipeline for the [Europarl-ST](https://www.mllp.upv.es/europarl-st/) dataset, converting raw speech translation data into training-ready Lhotse CutSet manifests with pre-extracted FBANK features.

## Overview

The pipeline transforms raw Europarl-ST data through the following stages:

```
Raw Europarl-ST v1.1
        │
        ▼
┌─────────────────┐
│  org_to_jsonl.py │  Extract audio segments + build per-language-pair JSONL
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  normalize_texts.py  │  Apply Whisper-style text normalization
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│ texts_to_cuts.py │  Generate Lhotse CutSet manifests + extract FBANK features
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│  filter_cuts_texts.py  │  Remove entries with empty text/st_text
└────────┬───────────────┘
         │
         ▼
┌──────────────────────┐
│  check_manifests.py  │  Validate final manifests
└──────────────────────┘

┌──────────────────┐
│  train_bpe.py    │  Train SentencePiece BPE model + generate tokens.txt
└──────────────────┘
```

## Prerequisites

### Optional

- [openai-whisper](https://github.com/openai/whisper) (`pip install openai-whisper`) — provides official text normalizers. If not installed, the scripts fall back to built-in local normalizers.

## Directory Structure

After running the full pipeline, the expected layout is:

```
Europarl-ST/
├── v1.1/                   # Raw Europarl-ST data (input)
│   ├── es/
│   ├── de/
│   ├── en/
│   └── ...
├── audio/                  # Extracted FLAC audio segments
│   ├── train/
│   ├── valid/
│   └── test/
├── texts/                  # Per-language-pair JSONL files
│   ├── es_de/
│   ├── es_en/
│   └── ...
├── normalizer/             # Normalized JSONL files
│   ├── es_de/
│   ├── es_en/
│   └── ...
├── fbank/                  # FBANK feature storage (.lca chunks)
│   └── feature_cache.json
├── manifests/              # Final Lhotse CutSet manifests
│   ├── es_de/
│   ├── es_en/
│   └── ...
├── bpe/                    # Trained BPE models and token lists
│   ├── asr9/              # ASR BPE (9-language shared)
│   │   ├── bpe.model
│   │   └── tokens.txt
│   ├── ast9/              # ST BPE (9-language shared)
│   │   ├── bpe.model
│   │   └── tokens.txt
│   └── ...
└── scripts/                # This directory
    ├── README.md
    ├── org_to_jsonl.py
    ├── normalize_texts.py
    ├── normalize_jsonl_with_whisper.py
    ├── filter_cuts_texts.py
    ├── texts_to_cuts.py
    ├── check_manifests.py
    └── train_bpe.py
```

## Scripts

### 1. `org_to_jsonl.py`

Extracts audio segments from raw Europarl-ST and produces per-language-pair JSONL files.

**What it does:**
- Iterates over all 9 languages (es, de, en, fr, nl, pl, pt, ro, it) and their pair combinations
- Cuts audio segments from `.m4a` files based on timestamps, converts to `.flac`
- Remaps dataset splits: `train` → `train`, `dev` → `valid`, `test` → `test`
- Writes JSONL entries with fields: `source`, `duration`, `text`, `st_text`

**Usage:**
```bash
python org_to_jsonl.py \
  --data-dir ../v1.1 \
  --output-dir ../audio
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `../v1.1` | Path to the raw Europarl-ST v1.1 directory |
| `--output-dir` | `../audio` | Directory for converted FLAC segments |

---

### 2. `normalize_texts.py`

Applies Whisper-style text normalization to the JSONL files produced by `org_to_jsonl.py`.

**What it does:**
- Normalizes Unicode (NFKC), replaces fancy quotes/dashes, removes control characters
- Automatically detects English fields from directory names (e.g., `es_en`) and applies the stricter English normalizer (retains only `[a-z0-9']`)
- Uses official `whisper.normalizers` if available, otherwise falls back to built-in implementations

**Usage:**
```bash
python normalize_texts.py \
  --src-dir ../texts \
  --dst-dir ../normalizer \
  --fields text st_text \
  --normalizer basic \
  --skip-existing
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--src-dir` | `../texts` | Root directory containing JSONL files |
| `--dst-dir` | `../texts/normalizer` | Destination for normalized output |
| `--fields` | `text st_text` | JSON keys to normalize (supports dot paths) |
| `--normalizer` | `basic` | Base normalizer: `basic` or `english` |
| `--no-auto-english` | (disabled) | Disable auto-detection of English fields |
| `--skip-existing` | (disabled) | Skip already-normalized files |
| `--dry-run` | (disabled) | Report stats without writing |
| `--verbose` | (disabled) | Enable DEBUG logging |

---

### 3. `normalize_jsonl_with_whisper.py`

Similar to `normalize_texts.py`, but operates on Lhotse CutSet-format `.jsonl.gz` files (post `texts_to_cuts.py`).

**What it does:**
- Normalizes `supervision.text` based on the `language` field
- Normalizes `supervision.custom.st_text` based on the `custom.lang` field
- Selects English normalizer for English fields, basic normalizer for others

**Usage:**
```bash
python normalize_jsonl_with_whisper.py \
  --input /path/to/input.jsonl.gz \
  --output /path/to/output.jsonl.gz
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--input` | Path to the source `.jsonl.gz` file |
| `--output` | Path to the destination `.jsonl.gz` file |
| `--keep-empty` | Keep empty lines from input (skipped by default) |

---

### 4. `texts_to_cuts.py`

Converts normalized JSONL files into Lhotse CutSet manifests with pre-extracted FBANK features.

**What it does:**
- Reads normalized JSONL, resolves audio paths, builds `Recording` and `SupervisionSegment` objects
- Extracts 80-dim FBANK features using `kaldifeat` (GPU-accelerated when available)
- Stores features in LilcomChunky (`.lca`) format for efficient random access
- Maintains a feature cache to avoid redundant computation across runs
- Supports sharding for large training sets

**Usage:**
```bash
python texts_to_cuts.py \
  --src-dir ../normalizer \
  --dst-dir ../manifests \
  --audio-root /path/to/audio/root \
  --storage-root ../fbank \
  --feature-cache ../fbank/feature_cache.json \
  --num-workers 8 \
  --batch-duration 600 \
  --skip-missing-audio \
  --verbose
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--src-dir` | `../texts/normalizer` | Normalized JSONL root directory |
| `--dst-dir` | `../cut_manifests` | Output directory for CutSet manifests |
| `--audio-root` | (parent of dataset dir) | Base path for resolving relative audio paths |
| `--storage-root` | `../fbank_storage` | Directory for `.lca` feature chunks |
| `--num-workers` | `8` | Workers for parallel feature extraction |
| `--batch-duration` | `600.0` | Total audio seconds per extraction batch |
| `--device` | `auto` | Device: `auto`, `cpu`, or `cuda` |
| `--train-shard-duration` | `0` (disabled) | Max audio seconds per train shard |
| `--skip-missing-audio` | (disabled) | Skip missing audio instead of raising |
| `--feature-cache` | (in storage root) | Path to feature cache JSON |
| `--refresh-cache` | (disabled) | Ignore cache, recompute all features |
| `--overwrite` | (disabled) | Overwrite existing outputs |
| `--verbose` | (disabled) | Enable DEBUG logging |

---

### 5. `filter_cuts_texts.py`

Removes CutSet entries that have empty `text` or `st_text` fields.

**What it does:**
- Scans all `*_cuts.jsonl` and `*_cuts.jsonl.gz` files in the manifest directory
- Removes any cut whose supervision has an empty `text` or `custom.st_text`
- Supports in-place overwrite or output to a separate directory

**Usage:**
```bash
python filter_cuts_texts.py \
  --manifest-dir ../manifests \
  --output-dir ../manifests \
  --overwrite \
  --verbose
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest-dir` | `../manifests` | Directory containing CutSet manifests |
| `--output-dir` | (same as manifest-dir) | Output directory for filtered manifests |
| `--overwrite` | (disabled) | Allow in-place replacement |
| `--dry-run` | (disabled) | Report stats without writing |
| `--verbose` | (disabled) | Enable DEBUG logging |

---

### 6. `check_manifests.py`

Validates the final CutSet manifests for correctness and data integrity.

**What it does:**
- Runs Lhotse's built-in `validate()` on all `*.jsonl.gz` manifests
- Optionally loads actual audio/features to detect stale metadata (`--read-data`)
- Reports per-cut failure counts and sample IDs (`--per-cut-details`)
- Supports multi-process parallel validation

**Usage:**
```bash
python check_manifests.py \
  --manifests-dir ../manifests \
  --num-workers 8 \
  --read-data \
  --verbose
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--manifests-dir` | `../manifests` | Directory containing manifests |
| `--pattern` | `*.jsonl.gz` | Glob pattern for files to validate |
| `--limit` | (none) | Cap number of files to inspect |
| `--num-workers` | `8` | Parallel worker processes |
| `--read-data` | (disabled) | Load audio/features (slow but thorough) |
| `--per-cut-details` | (disabled) | Report per-cut failure counts |
| `--bad-cut-samples` | `5` | Number of failing cut IDs to show |
| `--verbose` | (disabled) | Enable verbose logging |

### 7. `train_bpe.py`

Trains a SentencePiece BPE model from text transcripts and generates a `tokens.txt` vocabulary file.

**What it does:**
- Trains a unigram SentencePiece model with user-defined special symbols (blank, sos/eos, language tags)
- Generates `bpe.model` and `tokens.txt` in the specified output directory
- Supports custom vocabulary size for different tokenization granularities

**Usage:**
```bash
python train_bpe.py \
  --lang-dir ../Europarl-ST/bpe/ast9 \
  --transcript /path/to/training_text.txt \
  --vocab-size 6000
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--lang-dir` | Output directory for `bpe.model` and `tokens.txt` |
| `--transcript` | Path to training text file (one sentence per line) |
| `--vocab-size` | Target vocabulary size |

**Notes:**
- The model includes pre-defined language tags: `<2en>`, `<2de>`, `<2es>`, `<2fr>`, `<2it>`, `<2nl>`, `<2pl>`, `<2pt>`, `<2ro>`
- Special tokens `<blk>` and `<sos/eos>` are reserved
- Requires `sentencepiece >= 0.1.96` (`pip install sentencepiece`)

---

## Supported Languages

The dataset covers 9 European languages:

| Code | Language |
|------|----------|
| `es` | Spanish |
| `de` | German |
| `en` | English |
| `fr` | French |
| `nl` | Dutch |
| `pl` | Polish |
| `pt` | Portuguese |
| `ro` | Romanian |
| `it` | Italian |

All non-identical language pairs are processed (72 pairs total).

## Output Format

Each final CutSet manifest entry (MonoCut) contains:

```json
{
  "id": "es_1",
  "start": 0.0,
  "duration": 5.432,
  "channel": 0,
  "recording": { "...": "..." },
  "features": { "type": "kaldifeat-fbank", "...": "..." },
  "supervisions": [
    {
      "id": "es_1",
      "text": "source language transcription",
      "language": "es",
      "custom": {
        "st_text": "target language translation",
        "lang": "en"
      }
    }
  ]
}
```

## Quick Start

Run the full pipeline from the dataset root:

```bash
cd /path/to/Europarl-ST

# Step 1: Extract audio and build JSONL
python scripts/org_to_jsonl.py --data-dir ./v1.1 --output-dir ./audio

# Step 2: Normalize text
python scripts/normalize_texts.py --src-dir ./texts --dst-dir ./normalizer

# Step 3: Generate CutSet manifests with FBANK features
python scripts/texts_to_cuts.py \
  --src-dir ./normalizer \
  --dst-dir ./manifests \
  --audio-root /path/to/audio/root \
  --storage-root ./fbank \
  --num-workers 8 \
  --skip-missing-audio

# Step 4: Filter out entries with empty text
python scripts/filter_cuts_texts.py \
  --manifest-dir ./manifests \
  --overwrite

# Step 5: Validate
python scripts/check_manifests.py --manifests-dir ./manifests --read-data

# Step 6: Train BPE model
python scripts/train_bpe.py \
  --lang-dir ./bpe/ast9 \
  --transcript /path/to/all_training_text.txt \
  --vocab-size 6000
```

## License

Apache License 2.0
