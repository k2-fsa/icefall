# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                    Mingshuang Luo)
#
# See ../../LICENSE for clarification regarding multiple authors
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


import argparse
import collections
import logging
import os
import re
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union

import k2
import k2.version
import kaldialign
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from icefall.checkpoint import average_checkpoints

Pathlike = Union[str, Path]


# Pytorch issue: https://github.com/pytorch/pytorch/issues/47379
# Fixed: https://github.com/pytorch/pytorch/pull/49853
# The fix was included in v1.9.0
# https://github.com/pytorch/pytorch/releases/tag/v1.9.0
def is_jit_tracing():
    if torch.jit.is_scripting():
        return False
    elif torch.jit.is_tracing():
        return True
    return False


@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a
        # cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster() as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except Exception:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None


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
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


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


def encode_supervisions(
    supervisions: dict,
    subsampling_factor: int,
    token_ids: Optional[List[List[int]]] = None,
) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
    """
    Encodes Lhotse's ``batch["supervisions"]`` dict into
    a pair of torch Tensor, and a list of transcription strings or token indexes

    The supervision tensor has shape ``(batch_size, 3)``.
    Its second dimension contains information about sequence index [0],
    start frames [1] and num frames [2].

    The batch items might become re-ordered during this operation -- the
    returned tensor and list of strings are guaranteed to be consistent with
    each other.
    """
    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            torch.div(
                supervisions["start_frame"],
                subsampling_factor,
                rounding_mode="floor",
            ),
            torch.div(
                supervisions["num_frames"],
                subsampling_factor,
                rounding_mode="floor",
            ),
        ),
        1,
    ).to(torch.int32)

    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    if token_ids is None:
        texts = supervisions["text"]
        res = [texts[idx] for idx in indices]
    else:
        res = [token_ids[idx] for idx in indices]

    return supervision_segments, res


def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()


@dataclass
class DecodingResults:
    # timestamps[i][k] contains the frame number on which tokens[i][k]
    # is decoded
    timestamps: List[List[int]]

    # hyps[i] is the recognition results, i.e., word IDs or token IDs
    # for the i-th utterance with fast_beam_search_nbest_LG.
    hyps: Union[List[List[int]], k2.RaggedTensor]


def get_texts_with_timestamp(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> DecodingResults:
    """Extract the texts (as word IDs) and timestamps (as frame indexes)
    from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        all_aux_shape = (
            best_paths.arcs.shape().remove_axis(1).compose(best_paths.aux_labels.shape)
        )
        all_aux_labels = k2.RaggedTensor(all_aux_shape, best_paths.aux_labels.values)
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)
        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        all_aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = all_aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2

    timestamps = []
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        for p in range(all_aux_labels.dim0):
            time = []
            for i, arc in enumerate(all_aux_labels[p].tolist()):
                if len(arc) == 1 and arc[0] > 0:
                    time.append(i)
            timestamps.append(time)
    else:
        for labels in all_aux_labels.tolist():
            time = [i for i, v in enumerate(labels) if v > 0]
            timestamps.append(time)

    return DecodingResults(
        timestamps=timestamps,
        hyps=aux_labels if return_ragged else aux_labels.tolist(),
    )


def get_alignments(best_paths: k2.Fsa, kind: str) -> List[List[int]]:
    """Extract labels or aux_labels from the best-path FSAs.

    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      kind:
        Possible values are: "labels" and "aux_labels". Caution: When it is
        "labels", the resulting alignments contain repeats.
    Returns:
      Returns a list of lists of int, containing the token sequences we
      decoded. For `ans[i]`, its length equals to the number of frames
      after subsampling of the i-th utterance in the batch.

    Example:
      When `kind` is `labels`, one possible alignment example is (with
      repeats)::

        c c c blk a a blk blk t t t blk blk

     If `kind` is `aux_labels`, the above example changes to::

        c blk blk blk a blk blk blk t blk blk blk blk

    """
    assert kind in ("labels", "aux_labels")
    # arc.shape() has axes [fsa][state][arc], we remove "state"-axis here
    token_shape = best_paths.arcs.shape().remove_axis(1)
    # token_shape has axes [fsa][arc]
    tokens = k2.RaggedTensor(token_shape, getattr(best_paths, kind).contiguous())
    tokens = tokens.remove_values_eq(-1)
    return tokens.tolist()


def save_alignments(
    alignments: Dict[str, List[int]],
    subsampling_factor: int,
    filename: str,
) -> None:
    """Save alignments to a file.

    Args:
      alignments:
        A dict containing alignments. Keys of the dict are utterances and
        values are the corresponding framewise alignments after subsampling.
      subsampling_factor:
        The subsampling factor of the model.
      filename:
        Path to save the alignments.
    Returns:
      Return None.
    """
    ali_dict = {
        "subsampling_factor": subsampling_factor,
        "alignments": alignments,
    }
    torch.save(ali_dict, filename)


def load_alignments(filename: str) -> Tuple[int, Dict[str, List[int]]]:
    """Load alignments from a file.

    Args:
      filename:
        Path to the file containing alignment information.
        The file should be saved by :func:`save_alignments`.
    Returns:
      Return a tuple containing:
        - subsampling_factor: The subsampling_factor used to compute
          the alignments.
        - alignments: A dict containing utterances and their corresponding
          framewise alignment, after subsampling.
    """
    ali_dict = torch.load(filename)
    subsampling_factor = ali_dict["subsampling_factor"]
    alignments = ali_dict["alignments"]
    return subsampling_factor, alignments


def store_transcripts(
    filename: Pathlike, texts: Iterable[Tuple[str, str, str]]
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def store_transcripts_and_timestamps(
    filename: Pathlike,
    texts: Iterable[Tuple[str, List[str], List[str], List[float], List[float]]],
) -> None:
    """Save predicted results and reference transcripts as well as their timestamps
    to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp, time_ref, time_hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)
            if len(time_ref) > 0:
                s = "[" + ", ".join(["%0.3f" % i for i in time_ref]) + "]"
                print(f"{cut_id}:\ttimestamp_ref={s}", file=f)
            s = "[" + ", ".join(["%0.3f" % i for i in time_hyp]) + "]"
            print(f"{cut_id}:\ttimestamp_hyp={s}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
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
        An iterable of tuples. The first element is the cur_id, the second is
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
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
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


def write_error_stats_with_timestamps(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str], List[float], List[float]]],
    enable_log: bool = True,
) -> Tuple[float, float, float]:
    """Write statistics based on predicted results and reference transcripts
    as well as their timestamps.

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
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.

    Returns:
      Return total word error rate and mean delay.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    # Compute mean alignment delay on the correct words
    all_delay = []
    for cut_id, ref, hyp, time_ref, time_hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        has_time = len(time_ref) > 0 and len(time_hyp) > 0
        if has_time:
            # pointer to timestamp_hyp
            p_hyp = 0
            # pointer to timestamp_ref
            p_ref = 0
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
                if has_time:
                    p_hyp += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
                if has_time:
                    p_ref += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
                if has_time:
                    p_hyp += 1
                    p_ref += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
                if has_time:
                    all_delay.append(time_hyp[p_hyp] - time_ref[p_ref])
                    p_hyp += 1
                    p_ref += 1
        if has_time:
            assert p_hyp == len(hyp), (p_hyp, len(hyp))
            assert p_ref == len(ref), (p_ref, len(ref))

    ref_len = sum([len(r) for _, r, _, _, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    mean_delay = "inf"
    var_delay = "inf"
    num_delay = len(all_delay)
    if num_delay > 0:
        mean_delay = sum(all_delay) / num_delay
        var_delay = sum([(i - mean_delay) ** 2 for i in all_delay]) / num_delay
        mean_delay = "%.3f" % mean_delay
        var_delay = "%.3f" % var_delay

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )
        logging.info(
            f"[{test_set_name}] %symbol-delay mean: {mean_delay}s, variance: {var_delay} "  # noqa
            f"computed on {num_delay} correct words"
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
    for cut_id, ref, hyp, _, _ in results:
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
    return float(tot_err_rate), float(mean_delay), float(var_delay)


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


def concat(ragged: k2.RaggedTensor, value: int, direction: str) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      sos_id:
        The ID of the SOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")


def add_eos(ragged: k2.RaggedTensor, eos_id: int) -> k2.RaggedTensor:
    """Add EOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      eos_id:
        The ID of the EOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist ends with EOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_eos(a, eos_id=0)
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    return concat(ragged, eos_id, direction="right")


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

    expaned_lengths = torch.arange(max_len).expand(n, max_len).to(lengths)

    return expaned_lengths >= lengths.unsqueeze(1)


# Copied and modified from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/mask.py
def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def l1_norm(x):
    return torch.sum(torch.abs(x))


def l2_norm(x):
    return torch.sum(torch.pow(x, 2))


def linf_norm(x):
    return torch.max(torch.abs(x))


def measure_weight_norms(model: nn.Module, norm: str = "l2") -> Dict[str, float]:
    """
    Compute the norms of the model's parameters.

    :param model: a torch.nn.Module instance
    :param norm: how to compute the norm. Available values: 'l1', 'l2', 'linf'
    :return: a dict mapping from parameter's name to its norm.
    """
    with torch.no_grad():
        norms = {}
        for name, param in model.named_parameters():
            if norm == "l1":
                val = l1_norm(param)
            elif norm == "l2":
                val = l2_norm(param)
            elif norm == "linf":
                val = linf_norm(param)
            else:
                raise ValueError(f"Unknown norm type: {norm}")
            norms[name] = val.item()
        return norms


def measure_gradient_norms(model: nn.Module, norm: str = "l1") -> Dict[str, float]:
    """
    Compute the norms of the gradients for each of model's parameters.

    :param model: a torch.nn.Module instance
    :param norm: how to compute the norm. Available values: 'l1', 'l2', 'linf'
    :return: a dict mapping from parameter's name to its gradient's norm.
    """
    with torch.no_grad():
        norms = {}
        for name, param in model.named_parameters():
            if norm == "l1":
                val = l1_norm(param.grad)
            elif norm == "l2":
                val = l2_norm(param.grad)
            elif norm == "linf":
                val = linf_norm(param.grad)
            else:
                raise ValueError(f"Unknown norm type: {norm}")
            norms[name] = val.item()
        return norms


def optim_step_and_measure_param_change(
    model: nn.Module,
    old_parameters: Dict[str, nn.parameter.Parameter],
) -> Dict[str, float]:
    """
    Measure the "relative change in parameters per minibatch."
    It is understood as a ratio between the L2 norm of the difference between original and updates parameters,
    and the L2 norm of the original parameter. It is given by the formula:

        .. math::
            \begin{aligned}
                \delta = \frac{\Vert\theta - \theta_{new}\Vert^2}{\Vert\theta\Vert^2}
            \end{aligned}

    This function is supposed to be used as follows:

      .. code-block:: python

        old_parameters = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

        optimizer.step()

        deltas = optim_step_and_measure_param_change(old_parameters)

    Args:
      model: A torch.nn.Module instance.
      old_parameters:
        A Dict of named_parameters before optimizer.step().

    Return:
      A Dict containing the relative change for each parameter.
    """
    relative_change = {}
    with torch.no_grad():
        for n, p_new in model.named_parameters():
            p_orig = old_parameters[n]
            delta = l2_norm(p_orig - p_new) / l2_norm(p_orig)
            relative_change[n] = delta.item()
    return relative_change


def load_averaged_model(
    model_dir: str,
    model: torch.nn.Module,
    epoch: int,
    avg: int,
    device: torch.device,
):
    """
    Load a model which is the average of all checkpoints

    :param model_dir: a str of the experiment directory
    :param model: a torch.nn.Module instance

    :param epoch: the last epoch to load from
    :param avg: how many models to average from
    :param device: move model to this device

    :return: A model averaged
    """

    # start cannot be negative
    start = max(epoch - avg + 1, 0)
    filenames = [f"{model_dir}/epoch-{i}.pt" for i in range(start, epoch + 1)]

    logging.info(f"averaging {filenames}")
    model.to(device)
    model.load_state_dict(average_checkpoints(filenames, device=device))

    return model


def tokenize_by_bpe_model(
    sp: spm.SentencePieceProcessor,
    txt: str,
) -> str:
    """
    Tokenize text with bpe model. This function is from
    https://github1s.com/wenet-e2e/wenet/blob/main/wenet/dataset/processor.py#L322-L342.
    Args:
      sp: spm.SentencePieceProcessor.
      txt: str

    Return:
      A new string which includes chars and bpes.
    """
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r"([\u4e00-\u9fff])")
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)
    txt_with_bpe = "/".join(tokens)

    return txt_with_bpe


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def convert_timestamp(
    frames: List[int],
    subsampling_factor: int,
    frame_shift_ms: float = 10,
) -> List[float]:
    """Convert frame numbers to time (in seconds) given subsampling factor
    and frame shift (in milliseconds).

    Args:
      frames:
        A list of frame numbers after subsampling.
      subsampling_factor:
        The subsampling factor of the model.
      frame_shift_ms:
        Frame shift in milliseconds between two contiguous frames.
    Return:
      Return the time in seconds corresponding to each given frame.
    """
    frame_shift = frame_shift_ms / 1000.0
    time = []
    for f in frames:
        time.append(f * subsampling_factor * frame_shift)

    return time


def parse_timestamp(tokens: List[str], timestamp: List[float]) -> List[float]:
    """
    Parse timestamp of each word.

    Args:
      tokens:
        List of tokens.
      timestamp:
        List of timestamp of each token.

    Returns:
      List of timestamp of each word.
    """
    start_token = b"\xe2\x96\x81".decode()  # '_'
    assert len(tokens) == len(timestamp)
    ans = []
    for i in range(len(tokens)):
        flag = False
        if i == 0 or tokens[i].startswith(start_token):
            flag = True
            if len(tokens[i]) == 1 and tokens[i].startswith(start_token):
                # tokens[i] == start_token
                if i == len(tokens) - 1:
                    # it is the last token
                    flag = False
                elif tokens[i + 1].startswith(start_token):
                    # the next token also starts with start_token
                    flag = False
        if flag:
            ans.append(timestamp[i])
    return ans


def parse_hyp_and_timestamp(
    res: DecodingResults,
    subsampling_factor: int,
    frame_shift_ms: float = 10,
    sp: Optional[spm.SentencePieceProcessor] = None,
    word_table: Optional[k2.SymbolTable] = None,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Parse hypothesis and timestamp.

    Args:
      res:
        A DecodingResults object.
      subsampling_factor:
        The integer subsampling factor.
      frame_shift_ms:
        The float frame shift used for feature extraction.
      sp:
        The BPE model.
      word_table:
        The word symbol table.

    Returns:
       Return a list of hypothesis and timestamp.
    """
    hyps = []
    timestamps = []

    N = len(res.hyps)
    assert len(res.timestamps) == N, (len(res.timestamps), N)
    use_word_table = False
    if word_table is not None:
        assert sp is None
        use_word_table = True
    else:
        assert sp is not None and word_table is None

    for i in range(N):
        time = convert_timestamp(res.timestamps[i], subsampling_factor, frame_shift_ms)
        if use_word_table:
            words = [word_table[i] for i in res.hyps[i]]
        else:
            tokens = sp.id_to_piece(res.hyps[i])
            words = sp.decode_pieces(tokens).split()
            time = parse_timestamp(tokens, time)
        assert len(time) == len(words), (len(time), len(words))

        hyps.append(words)
        timestamps.append(time)

    return hyps, timestamps


# `is_module_available` is copied from
# https://github.com/pytorch/audio/blob/6bad3a66a7a1c7cc05755e9ee5931b7391d2b94c/torchaudio/_internal/module_utils.py#L9
def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`.

    Note: "borrowed" from torchaudio:
    """
    import importlib

    return all(importlib.util.find_spec(m) is not None for m in modules)
