#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Common Voice manifests for the Japanese KWS Zipformer trainer."""

import argparse
import logging
import sys
from pathlib import Path

# Import the maintained Common Voice data module by its package name.  A
# file-path import gives its classes a synthetic module name, which Python
# 3.14's default ``forkserver`` multiprocessing context cannot import again
# when starting DataLoader workers.
ICEFALL_ROOT = Path(__file__).resolve().parents[4]
if str(ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ICEFALL_ROOT))

from egs.commonvoice.ASR.pruned_transducer_stateless7_streaming import (  # noqa: E402
    asr_datamodule as module,
)

# Common Voice occasionally contains multi-paragraph transcripts paired with
# only a few seconds of audio.  These are alignment errors rather than fast
# speech and make RNN-T memory scale with thousands of target symbols.
MAX_TRANSCRIPT_CHARS_PER_SECOND = 20.0


def has_plausible_transcript_rate(cut) -> bool:
    text = cut.supervisions[0].text
    return len(text) / cut.duration <= MAX_TRANSCRIPT_CHARS_PER_SECOND


class CommonVoiceKwsDataModule(module.CommonVoiceAsrDataModule):
    """Reuse the maintained Common Voice loader with KWS naming semantics."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)

    def train_cuts(self):
        logging.info(
            "Exclude train cuts above %.1f transcript characters/second",
            MAX_TRANSCRIPT_CHARS_PER_SECOND,
        )
        return super().train_cuts().filter(has_plausible_transcript_rate)

    def valid_cuts(self):
        """The Common Voice development split is this recipe's validation set."""

        logging.info(
            "Exclude dev cuts above %.1f transcript characters/second",
            MAX_TRANSCRIPT_CHARS_PER_SECOND,
        )
        return self.dev_cuts().filter(has_plausible_transcript_rate)


# The maintained WenetSpeech KWS trainer imports this exact name at module
# import time.  Keep the adapter compatible before train.py replaces the
# reference with the explicit CommonVoiceKwsDataModule name.
WenetSpeechAsrDataModule = CommonVoiceKwsDataModule
