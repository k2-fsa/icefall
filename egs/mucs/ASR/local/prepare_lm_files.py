#!/usr/bin/env python3

import argparse
import gzip
import logging
import os
import shutil
from pathlib import Path

from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, help="Output directory.")
    parser.add_argument("--data-path", type=str, help="Input directory.")
    parser.add_argument("--mode", type=str, help="Input split")
    args = parser.parse_args()
    return args

def read_text(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return [' '.join(l.split(' ')[1:]) for l in lines]

def create_files(text):
    lexicon = {}
    for line in text:
        for word in line.split(' '):
            if word.strip() == '': continue
            if word not in lexicon:
                lexicon[word] = ' '.join(list(word))
    with open(os.path.join(args.out_dir, 'mucs_lexicon.txt'), 'w') as f:
        for word in lexicon:
            f.write(word + '\t' + lexicon[word] + '\n')
    with open(os.path.join(args.out_dir, 'mucs_vocab.txt'), 'w') as f:
        for word in lexicon:
            f.write(word + '\n')
    with open(os.path.join(args.out_dir, 'mucs_vocab_text.txt'), 'w') as f:
        for line in text:
            f.write(line + '\n')
            
def main():
    path = os.path.join(args.data_path, args.mode)
    text = read_text(os.path.join(path, "text"))
    create_files(text)
    
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(f"out_dir: {args.out_dir}")
    logging.info(f"in_dir: {args.data_path}")
    main()
