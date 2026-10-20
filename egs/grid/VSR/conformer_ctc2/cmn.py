# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""Cepstral mean normalisation for AV-HuBERT features.

Kaldi's ``apply-cmvn`` (mean only, which is its default) subtracts a mean from
every feature dimension before the acoustic model sees it. The ssl-kaldi GRID
recipe does this per speaker; single-clip inference (lipreader.py) has no
speaker labels, so the equivalent here is per utterance.

Measured on layer-4 mean-face features, the domain shift from GRID to a
second frontal corpus is almost entirely a mean offset: the mean moves by 73.8 in
feature space, three times the typical within-GRID per-utterance deviation
(24.4), while per-dimension standard deviations are unchanged (median ratio
0.99). Subtracting the per-utterance mean removes that offset, so variance
normalisation is not needed and is not done.

The same mode must be used for training, decoding and lipreader.py: a model
trained on normalised features decodes unnormalised ones poorly, and the
mismatch is silent.
"""
import torch

MODES = ("none", "utt")


def apply_cmn(
    feature: torch.Tensor, mode: str = "none", lengths: torch.Tensor = None
) -> torch.Tensor:
    """Mean-normalise features.

    Parameters
    ----------
    feature : (N, T, C) batch or (T, C) single utterance.
    mode : "none" (identity) or "utt" (subtract each utterance's own mean).
    lengths : optional (N,) valid frame counts. Padded frames are excluded
        from the mean, which otherwise pulls it toward whatever the padding
        holds.
    """
    if mode == "none":
        return feature
    if mode != "utt":
        raise ValueError(f"unknown cmn mode: {mode!r} (expected one of {MODES})")

    if feature.ndim == 2:  # (T, C)
        return feature - feature.mean(dim=0, keepdim=True)

    if feature.ndim != 3:
        raise ValueError(f"expected (N, T, C) or (T, C), got {tuple(feature.shape)}")

    if lengths is None:
        return feature - feature.mean(dim=1, keepdim=True)

    # Masked mean over the valid frames of each utterance.
    n, t, _ = feature.shape
    mask = (
        torch.arange(t, device=feature.device)[None, :]
        < lengths.to(feature.device)[:, None]
    )
    mask = mask.unsqueeze(-1).to(feature.dtype)  # (N, T, 1)
    total = (feature * mask).sum(dim=1, keepdim=True)  # (N, 1, C)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (N, 1, 1)
    return feature - total / count
