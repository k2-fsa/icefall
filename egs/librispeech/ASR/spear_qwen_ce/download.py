#!/usr/bin/env python3
"""Download SPEAR-Qwen3 CE checkpoints and their Qwen3 base model."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = 'yfyeung/icefall-asr-librispeech-spear-qwen3-ce'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=str.upper, choices=['0.6B', '1.7B', '4B', '8B', '14B', '32B'], required=True)
    parser.add_argument('--decoding-method', choices=['greedy_search', 'beam_search', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('download/spear-qwen3-ce'))
    parser.add_argument('--repo-id', default=REPO_ID)
    parser.add_argument('--skip-base', action='store_true', help='Use an existing Qwen3 base model')
    args = parser.parse_args()
    hf_hub_download(args.repo_id, 'release.json', local_dir=args.output_dir)
    release = json.loads((args.output_dir / 'release.json').read_text())
    model = release['models'][args.size]
    methods = ['greedy_search', 'beam_search'] if args.decoding_method == 'all' else [args.decoding_method]
    checkpoints = {model[method]['checkpoint'] for method in methods}
    patterns = ['release.json', 'README.md', 'LICENSE',
                'spear_qwen_ce/*', 'exp/*.sh', 'encoder/*',
                f"{model['base_model']['local_dir']}/*", *checkpoints]
    snapshot_download(args.repo_id, local_dir=args.output_dir, allow_patterns=patterns)
    if not args.skip_base:
        base = model['base_model']
        snapshot_download(base['repo_id'], local_dir=args.output_dir / base['local_dir'],
                          allow_patterns=['*.safetensors', '*.json', '*.txt'])
    print(f'Ready: {args.size} in {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
