# https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/utils/get_random_segments.py

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Function to get random segments."""

from typing import Any, Dict, List, Optional, Tuple, Union
import collections
import logging
import re
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from lhotse.dataset.sampling.base import CutSampler
from pathlib import Path
from phonemizer import phonemize
from symbols import symbol_table
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from unidecode import unidecode


def get_random_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get random segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        x_lengths (Tensor): Length tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
        Tensor: Start index tensor (B,).

    """
    b, c, t = x.size()
    max_start_idx = x_lengths - segment_size
    max_start_idx[max_start_idx < 0] = 0
    start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
        dtype=torch.long,
    )
    segments = get_segments(x, start_idxs, segment_size)

    return segments, start_idxs


def get_segments(
    x: torch.Tensor,
    start_idxs: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """Get segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        start_idxs (Tensor): Start index tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).

    """
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx : start_idx + segment_size]
    return segments


# https://github.com/espnet/espnet/blob/master/espnet2/torch_utils/device_funcs.py
def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively

    The difference from to_device() is changing to torch.Tensor if float or int
    value is found.

    The restriction to the returned value in DataParallel:
        The object must be
        - torch.cuda.Tensor
        - 1 or more dimension. 0-dimension-tensor sends warning.
        or a list, tuple, dict.

    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # DataParallel can't handle NamedTuple well
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:
            # To 1-dim array
            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        warnings.warn(f"{type(data)} may not be gatherable by DataParallel")
        return data


# The following codes are based on https://github.com/jaywalnut310/vits

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def text_clean(text):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress.

    Returns:
        A string of phonemes.
    '''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language='en-us',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes


# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbol_table)}
id_to_symbol = {i: s for i, s in enumerate(symbol_table)}


# def text_to_sequence(text: str) -> List[int]:
#     '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
#     '''
#     cleaned_text = text_clean(text)
#     sequence = [symbol_to_id[symbol] for symbol in cleaned_text]
#     return sequence
#
#
# def sequence_to_text(sequence: List[int]) -> str:
#     '''Converts a sequence of IDs back to a string'''
#     result = ''.join(id_to_symbol[symbol_id] for symbol_id in sequence)
#     return result


def intersperse(sequence, item=0):
    result = [item] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result


def prepare_token_batch(
    texts: List[str],
    phonemes: Optional[List[str]] = None,
    intersperse_blank: bool = True,
    blank_id: int = 0,
    pad_id: int = 0,
) -> torch.Tensor:
    """Convert a list of text strings into a batch of symbol tokens with padding.
    Args:
        texts: list of text strings
        intersperse_blank: whether to intersperse blank tokens in the converted token sequence.
        blank_id: index of blank token
        pad_id: padding index
    """
    if phonemes is None:
        # normalize text
        normalized_texts = []
        for text in texts:
            text = convert_to_ascii(text)
            text = lowercase(text)
            text = expand_abbreviations(text)
            normalized_texts.append(text)

        # convert to phonemes
        phonemes = phonemize(
            normalized_texts,
            language='en-us',
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        phonemes = [collapse_whitespace(sequence) for sequence in phonemes]

    # convert to symbol ids
    lengths = []
    sequences = []
    skip = False
    for idx, sequence in enumerate(phonemes):
        try:
            sequence = [symbol_to_id[symbol] for symbol in sequence]
        except Exception:
            # print(texts[idx])
            # print(normalized_texts[idx])
            print(phonemes[idx])
            skip = True
        if intersperse_blank:
            sequence = intersperse(sequence, blank_id)
        try:
            sequences.append(torch.tensor(sequence, dtype=torch.int64))
        except Exception:
            print(sequence)
            skip = True
        lengths.append(len(sequence))

    sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return sequences, lengths, skip


class MetricsTracker(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            ans += str(k) + "=" + str(norm_value) + ", "
        samples = "%.2f" % self["samples"]
        ans += "over " + str(samples) + " samples."
        return ans

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('loss_1', 0.1), ('loss_2', 0.07)]
        """
        samples = self["samples"] if "samples" in self else 1
        ans = []
        for k, v in self.items():
            if k == "samples":
                continue
            norm_value = float(v) / samples
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)


# checkpoint saving and loading
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def save_checkpoint(
    filename: Path,
    model: Union[nn.Module, DDP],
    params: Optional[Dict[str, Any]] = None,
    optimizer_g: Optional[Optimizer] = None,
    optimizer_d: Optional[Optimizer] = None,
    scheduler_g: Optional[LRSchedulerType] = None,
    scheduler_d: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    rank: int = 0,
) -> None:
    """Save training information to a file.

    Args:
      filename:
        The checkpoint filename.
      model:
        The model to be saved. We only save its `state_dict()`.
      model_avg:
        The stored model averaged from the start of training.
      params:
        User defined parameters, e.g., epoch, loss.
      optimizer_g:
        The optimizer for generator used in the training.
        Its `state_dict` will be saved.
      optimizer_d:
        The optimizer for discriminator used in the training.
        Its `state_dict` will be saved.
      scheduler_g:
        The learning rate scheduler for generator used in the training.
        Its `state_dict` will be saved.
      scheduler_d:
        The learning rate scheduler for discriminator used in the training.
        Its `state_dict` will be saved.
      scalar:
        The GradScaler to be saved. We only save its `state_dict()`.
      rank:
        Used in DDP. We save checkpoint only for the node whose rank is 0.
    Returns:
      Return None.
    """
    if rank != 0:
        return

    logging.info(f"Saving checkpoint to {filename}")

    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
        "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    torch.save(checkpoint, filename)


def save_checkpoint_with_global_batch_idx(
    out_dir: Path,
    global_batch_idx: int,
    model: Union[nn.Module, DDP],
    params: Optional[Dict[str, Any]] = None,
    optimizer_g: Optional[Optimizer] = None,
    optimizer_d: Optional[Optimizer] = None,
    scheduler_g: Optional[LRSchedulerType] = None,
    scheduler_d: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    rank: int = 0,
):
    """Save training info after processing given number of batches.

    Args:
      out_dir:
        The directory to save the checkpoint.
      global_batch_idx:
        The number of batches processed so far from the very start of the
        training. The saved checkpoint will have the following filename:
          f'out_dir / checkpoint-{global_batch_idx}.pt'
      model:
        The neural network model whose `state_dict` will be saved in the
        checkpoint.
      params:
        A dict of training configurations to be saved.
      optimizer_g:
        The optimizer for generator used in the training.
        Its `state_dict` will be saved.
      optimizer_d:
        The optimizer for discriminator used in the training.
        Its `state_dict` will be saved.
      scheduler_g:
        The learning rate scheduler for generator used in the training.
        Its `state_dict` will be saved.
      scheduler_d:
        The learning rate scheduler for discriminator used in the training.
        Its `state_dict` will be saved.
      scaler:
        The scaler used for mix precision training. Its `state_dict` will
        be saved.
      sampler:
        The sampler used in the training dataset.
      rank:
        The rank ID used in DDP training of the current node. Set it to 0
        if DDP is not used.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"checkpoint-{global_batch_idx}.pt"
    save_checkpoint(
        filename=filename,
        model=model,
        params=params,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        scaler=scaler,
        sampler=sampler,
        rank=rank,
    )


# def plot_feature(feature):
#     """
#     Display the feature matrix as an image. Requires matplotlib to be installed.
#     """
#     import matplotlib.pyplot as plt
#
#     feature = np.flip(feature.transpose(1, 0), 0)
#     return plt.matshow(feature)

MATPLOTLIB_FLAG = False


def plot_feature(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
