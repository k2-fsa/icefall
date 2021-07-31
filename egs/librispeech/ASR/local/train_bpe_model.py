#!/usr/bin/env python3

"""
This script takes as input "data/lang/bpe/train.txt"
and generates "data/lang/bpe/bep.model".
"""

# You can install sentencepiece via:
#
#  pip install sentencepiece
#
# Due to an issue reported in
# https://github.com/google/sentencepiece/pull/642#issuecomment-857972030
#
# Please install a version >=0.1.96

from pathlib import Path

import sentencepiece as spm

import shutil


def main():
    model_type = "unigram"
    vocab_size = 5000
    model_prefix = f"data/lang/bpe/{model_type}_{vocab_size}"
    train_text = "data/lang/bpe/train.txt"
    character_coverage = 1.0
    input_sentence_size = 100000000

    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
            unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )

    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    vocab_size = sp.vocab_size()

    shutil.copyfile(model_file, "data/lang/bpe/bpe.model")


if __name__ == "__main__":
    main()
