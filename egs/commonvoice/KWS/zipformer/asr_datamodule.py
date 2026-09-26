#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Common Voice manifests for the Japanese KWS Zipformer trainer."""

import argparse
import importlib.util
from pathlib import Path


COMMONVOICE_ASR_DATA_MODULE = (
    Path(__file__).resolve().parents[2]
    / "ASR"
    / "pruned_transducer_stateless7_streaming"
    / "asr_datamodule.py"
)
spec = importlib.util.spec_from_file_location(
    "commonvoice_asr_datamodule", COMMONVOICE_ASR_DATA_MODULE
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {COMMONVOICE_ASR_DATA_MODULE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CommonVoiceKwsDataModule(module.CommonVoiceAsrDataModule):
    """Reuse the maintained Common Voice loader with KWS naming semantics."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)

    def valid_cuts(self):
        """The Common Voice development split is this recipe's validation set."""

        return self.dev_cuts()
