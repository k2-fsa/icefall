#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Train the maintained tiny KWS Zipformer on Japanese phone tokens.

The WenetSpeech KWS trainer contains the streaming model, pruned RNN-T loss,
checkpoint, and distributed-training mechanics. This adapter only replaces its
dataset loader and Chinese pinyin frontend; it avoids an unreviewable fork of
the training loop while preserving the upstream trainer behavior.
"""

import importlib.util
import logging
from pathlib import Path
import sys

ICEFALL_ROOT = Path(__file__).resolve().parents[4]
if str(ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ICEFALL_ROOT))

import k2

# Use the fully qualified recipe module here.  The upstream WenetSpeech
# trainer is loaded below and intentionally adds its own directory to
# ``sys.path``; importing this adapter as plain ``asr_datamodule`` would make
# Python 3.14's forkserver workers resolve the wrong module when they rerun
# this entrypoint.
from egs.commonvoice.KWS.zipformer.asr_datamodule import (  # noqa: E402
    CommonVoiceKwsDataModule,
)


KWS_LOCAL = Path(__file__).resolve().parents[1] / "local"
if str(KWS_LOCAL) not in sys.path:
    sys.path.insert(0, str(KWS_LOCAL))

from japanese_phones import text_to_phones


WENETSPEECH_ZIPFORMER = (
    Path(__file__).resolve().parents[3] / "wenetspeech" / "KWS" / "zipformer"
)
if str(WENETSPEECH_ZIPFORMER) not in sys.path:
    sys.path.insert(0, str(WENETSPEECH_ZIPFORMER))

spec = importlib.util.spec_from_file_location(
    "wenetspeech_kws_train", WENETSPEECH_ZIPFORMER / "train.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {WENETSPEECH_ZIPFORMER / 'train.py'}")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def encode_text(cut, token_table: k2.SymbolTable, params):
    """Attach Japanese phone IDs to a cut for the upstream RNN-T trainer."""

    text = cut.supervisions[0].text
    phones = text_to_phones(text)
    ids = []
    for phone in phones:
        if phone in token_table:
            ids.append(token_table[phone])
        else:
            logging.warning("Text %r has OOV phone %r; using <unk>", text, phone)
            ids.append(token_table["<unk>"])
    cut.supervisions[0].tokens = ids
    return cut


trainer.WenetSpeechAsrDataModule = CommonVoiceKwsDataModule
trainer.encode_text = encode_text


if __name__ == "__main__":
    trainer.main()
