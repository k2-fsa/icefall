import argparse
import collections
import json
import logging
import os
import pathlib
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union
from tqdm import tqdm
import kaldialign
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
Pathlike = Union[str, Path]


def get_world_size():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_local_rank()
    else:
        return 0


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

    def __str__(self, indent: int = 2):
        tmp = {}
        for k, v in self.items():
            # PosixPath is ont JSON serializable
            if isinstance(v, pathlib.Path) or isinstance(v, torch.device):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)


def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


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
            if v - v == 0:
                ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
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


def store_transcripts(
    filename: Pathlike, texts: Iterable[Tuple[str, str, str]], char_level: bool = False
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
        If it is a multi-talker ASR system, the ref and hyp may also be lists of
        strings.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf8") as f:
        for cut_id, ref, hyp in texts:
            if char_level:
                ref = list("".join(ref))
                hyp = list("".join(hyp))
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
    compute_CER: bool = False,
    sclite_mode: bool = False,
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cut_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"

    if compute_CER:
        for i, res in enumerate(results):
            cut_id, ref, hyp = res
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results[i] = (cut_id, ref, hyp)

    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate)