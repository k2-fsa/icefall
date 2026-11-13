# Copyright    2026  Shanghai Jiao Tong University  (authors: Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load the SPEAR speech encoder (a local ``trust_remote_code`` HF snapshot).

transformers 4.57's dynamic-module loader doesn't copy a local custom-code
model's sibling ``.py`` files, so ``from_pretrained(local_dir,
trust_remote_code=True)`` dies with ``FileNotFoundError`` on relative imports.
Workaround: register the model dir as a synthetic package, exec its modules in
dependency order, then ``SpearModel.from_pretrained`` -- no dynamic cache.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

_PKG = "spear_remote_code"
# Dependency order: each module's relative imports must resolve to an already
# loaded sibling. modeling_spear -> spear_model -> {zipformer -> spear_modules,
# configuration_spear}.
_MODULES = (
    "configuration_spear",
    "spear_modules",
    "zipformer",
    "spear_model",
    "modeling_spear",
)


def load_spear_encoder(model_dir: str) -> nn.Module:
    """Load a SPEAR encoder from a local HF snapshot ``model_dir``.

    The returned module bundles a kaldi-fbank with a Zipformer; this recipe drives
    only the latter, via ``spear_encode_fbank``.
    """
    model_dir = Path(model_dir).resolve()
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(model_dir)]
        sys.modules[_PKG] = pkg

    for name in _MODULES:
        full = f"{_PKG}.{name}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full, str(model_dir / f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)

    modeling = sys.modules[f"{_PKG}.modeling_spear"]
    model_cls = modeling.SpearModel
    # SPEAR has no tied weights but transformers >=4.57.6 reads
    # all_tied_weights_keys on load; default to {}.
    if not hasattr(model_cls, "all_tied_weights_keys"):
        model_cls.all_tied_weights_keys = {}
    model = model_cls.from_pretrained(str(model_dir))

    # RelPositionalEncoding.pe is a plain (non-buffer) attr rebuilt lazily in
    # forward, so it's absent from the checkpoint; under the DLC image's
    # accelerate meta-init it's left on the meta device and extend_pe then crashes
    # ("Cannot copy out of meta tensor"). Drop it so forward rebuilds it for real.
    for m in model.modules():
        pe = getattr(m, "pe", None)
        if torch.is_tensor(pe) and pe.is_meta:
            m.pe = None
    return model


def spear_encoder_dim(encoder: nn.Module) -> int:
    """Top encoder output dim (max over SPEAR's per-stack encoder dims)."""
    return max(int(d) for d in str(encoder.config.encoder_dim).split(","))


# Entry point for the Zipformer half of SPEAR's forward, skipping the kaldi-fbank
# it bundles in front: lhotse computes an equivalent fbank in the dataloader (see
# ``asr_datamodule``), so SPEAR's own per-utterance loop would be redundant work
# on the training critical path. ``encoder`` is the HF ``SpearModel`` from
# ``load_spear_encoder`` (``.model`` is the inner SpearModel, ``.model.model`` its
# SpearEncoder).
def spear_encode_fbank(
    encoder: nn.Module,
    fbank: torch.Tensor,
    fbank_lens: torch.Tensor,
    return_middle_layers: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run SPEAR's Zipformer over a (possibly SpecAugment'd) 128-bin fbank,
    returning the same dict as ``encoder``: ``encoder_out`` (N, T', dim),
    ``encoder_out_lens`` and ``hidden_states``."""
    return encoder.model.model.forward_encoder(
        fbank, fbank_lens, return_middle_out=return_middle_layers, return_dict=True
    )
