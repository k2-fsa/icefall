from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")


def prepare_input(
    params: AttributeDict,
    batch: dict,
    device: torch.device,
    tokenizer: Optional[Any] = None,
    return_tokens: bool = False,
    return_feature: bool = False,
    return_audio: bool = False,
    return_prompt: bool = False,
):
    """
    Parse the features and targets of the current batch.
    Args:
      params:
        It is returned by :func:`get_params`.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      sp:
        Used to convert text to bpe tokens.
      device:
        The device of Tensor.
    """
    return_list = []

    if return_tokens:
        assert tokenizer is not None

        if params.token_type == "phone":
            tokens = tokenizer.tokens_to_token_ids(batch["tokens"])
        else:
            tokens = tokenizer.texts_to_token_ids(batch["text"])
        return_list += [tokens]

    if return_feature:
        features = batch["features"].to(device)
        features_lens = batch["features_lens"].to(device)
        return_list += [features * params.feat_scale, features_lens]

    if return_audio:
        return_list += [batch["audio"], batch["audio_lens"]]

    if return_prompt:
        if return_tokens:
            if params.token_type == "phone":
                prompt_tokens = tokenizer.tokens_to_token_ids(batch["prompt"]["tokens"])
            else:
                prompt_tokens = tokenizer.texts_to_token_ids(batch["prompt"]["text"])
            return_list += [prompt_tokens]
        if return_feature:
            prompt_features = batch["prompt"]["features"].to(device)
            prompt_features_lens = batch["prompt"]["features_lens"].to(device)
            return_list += [prompt_features * params.feat_scale, prompt_features_lens]
        if return_audio:
            return_list += [batch["prompt"]["audio"], batch["prompt"]["audio_lens"]]

    return return_list


def prepare_avg_tokens_durations(features_lens, tokens_lens):
    tokens_durations = []
    for i in range(len(features_lens)):
        utt_duration = features_lens[i]
        avg_token_duration = utt_duration // tokens_lens[i]
        tokens_durations.append([avg_token_duration] * tokens_lens[i])
    return tokens_durations


def pad_labels(y: List[List[int]], pad_id: int, device: torch.device):
    """
    Pad the transcripts to the same length with zeros.

    Args:
      y: the transcripts, which is a list of a list

    Returns:
      Return a Tensor of padded transcripts.
    """
    y = [l + [pad_id] for l in y]
    length = max([len(l) for l in y])
    y = [l + [pad_id] * (length - len(l)) for l in y]
    return torch.tensor(y, dtype=torch.int64, device=device)


def get_tokens_index(durations: List[List[int]], num_frames: int) -> torch.Tensor:
    """
    Gets position in the transcript for each frame, i.e. the position
    in the symbol-sequence to look up.

    Args:
      durations:
        Duration of each token in transcripts.
      num_frames:
        The maximum frame length of the current batch.

    Returns:
      Return a Tensor of shape (batch_size, num_frames)
    """
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        assert cur_frame == num_frames, (cur_frame, num_frames)
    return ans


def to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def condition_time_mask(
    features_lens: torch.Tensor, mask_percent: Tuple[float, float], max_len: int = 0
) -> torch.Tensor:
    """
    Apply Time masking.
    Args:
        features_lens:
            input tensor of shape ``(B)``
        mask_size:
            the width size for masking.
        max_len:
            the maximum length of the mask.
    Returns:
        Return a 2-D bool tensor (B, T), where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.
    """
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (
        seq_range[None, :] < mask_ends[:, None]
    )
    return mask


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)
