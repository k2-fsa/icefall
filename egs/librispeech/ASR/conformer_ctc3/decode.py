#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
"""
Usage:
(1) decode in non-streaming mode (take ctc-decoding as an example)
./conformer_ctc3/decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./conformer_ctc3/exp \
    --max-duration 600 \
    --decoding-method ctc-decoding

(2) decode in streaming mode (take ctc-decoding as an example)
./conformer_ctc3/decode.py \
    --epoch 30 \
    --avg 15 \
    --simulate-streaming 1 \
    --causal-convolution 1 \
    --decode-chunk-size 16 \
    --left-context 64 \
    --exp-dir ./conformer_ctc3/exp \
    --max-duration 600 \
    --decoding-method ctc-decoding

To evaluate symbol delay, you should:
(1) Generate cuts with word-time alignments:
./add_alignments.sh
(2) Set the argument "--manifest-dir data/fbank_ali" while decoding.
For example:
./conformer_ctc3/decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./conformer_ctc3/exp \
    --max-duration 600 \
    --decoding-method ctc-decoding \
    --simulate-streaming 1 \
    --causal-convolution 1 \
    --decode-chunk-size 16 \
    --left-context 64 \
    --manifest-dir data/fbank_ali
Note: It supports calculating symbol delay with following decoding methods:
    - ctc-decoding
    - 1best
"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from train import add_model_arguments, get_ctc_model, get_params

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    convert_timestamp,
    get_texts,
    make_pad_mask,
    parse_bpe_start_end_pairs,
    parse_fsa_timestamps_and_texts,
    setup_logger,
    store_transcripts_and_timestamps,
    str2bool,
    write_error_stats_with_timestamps,
)

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless4/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="ctc-decoding",
        help="""Decoding method.
        Supported values are:
        - (0) ctc-greedy-search. It uses a sentence piece model,
          i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
        - (1) ctc-decoding. Use CTC decoding. It uses a sentence piece
          model, i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
        - (2) 1best. Extract the best path from the decoding lattice as the
          decoding result.
        - (3) nbest. Extract n paths from the decoding lattice; the path
          with the highest score is the decoding result.
        - (4) nbest-rescoring. Extract n paths from the decoding lattice,
          rescore them with an n-gram LM (e.g., a 4-gram LM), the path with
          the highest score is the decoding result.
        - (5) whole-lattice-rescoring. Rescore the decoding lattice with an
          n-gram LM (e.g., a 4-gram LM), the best path of rescored lattice
          is the decoding result.
          you have trained an RNN LM using ./rnn_lm/train.py
        - (6) nbest-oracle. Its WER is the lower bound of any n-best
          rescoring method can achieve. Useful for debugging n-best
          rescoring method.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="""The n-gram LM dir.
        It should contain either G_4_gram.pt or G_4_gram.fst.txt
        """,
    )

    parser.add_argument(
        "--simulate-streaming",
        type=str2bool,
        default=False,
        help="""Whether to simulate streaming in decoding, this is a good way to
        test a streaming model.
        """,
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=16,
        help="The chunk size for decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--left-context",
        type=int,
        default=64,
        help="left context can be seen during decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--hlg-scale",
        type=float,
        default=0.8,
        help="""The scale to be applied to `hlg.scores`.
        """,
    )

    add_model_arguments(parser)

    return parser


def get_decoding_params() -> AttributeDict:
    """Parameters for decoding."""
    params = AttributeDict(
        {
            "frame_shift_ms": 10,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def ctc_greedy_search(
    ctc_probs: torch.Tensor,
    nnet_output_lens: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    subsampling_factor: int = 4,
    frame_shift_ms: float = 10,
) -> Tuple[List[Tuple[float, float]], List[List[str]]]:
    """Apply CTC greedy search
    Args:
      ctc_probs (torch.Tensor):
        (batch, max_len, feat_dim)
      nnet_output_lens (torch.Tensor):
        (batch, )
      sp:
        The BPE model.
      subsampling_factor:
        The subsampling factor of the model.
      frame_shift_ms:
        Frame shift in milliseconds between two contiguous frames.

    Returns:
      utt_time_pairs:
        A list of pair list. utt_time_pairs[i] is a list of
        (start-time, end-time) pairs for each word in
        utterance-i.
      utt_words:
        A list of str list. utt_words[i] is a word list of utterence-i.
    """
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.squeeze(2)  # (B, maxlen)
    mask = make_pad_mask(nnet_output_lens)
    topk_index = topk_index.masked_fill_(mask, 0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]

    def get_first_tokens(tokens: List[int]) -> List[bool]:
        is_first_token = []
        first_tokens = []
        for t in range(len(tokens)):
            if tokens[t] != 0 and (t == 0 or tokens[t - 1] != tokens[t]):
                is_first_token.append(True)
                first_tokens.append(tokens[t])
            else:
                is_first_token.append(False)
        return first_tokens, is_first_token

    utt_time_pairs = []
    utt_words = []
    for utt in range(len(hyps)):
        first_tokens, is_first_token = get_first_tokens(hyps[utt])
        all_tokens = sp.id_to_piece(hyps[utt])
        index_pairs = parse_bpe_start_end_pairs(all_tokens, is_first_token)
        words = sp.decode(first_tokens).split()
        assert len(index_pairs) == len(words), (
            len(index_pairs),
            len(words),
            all_tokens,
        )
        start = convert_timestamp(
            frames=[i[0] for i in index_pairs],
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )
        end = convert_timestamp(
            # The duration in frames is (end_frame_index - start_frame_index + 1)
            frames=[i[1] + 1 for i in index_pairs],
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )
        utt_time_pairs.append(list(zip(start, end)))
        utt_words.append(words)

    return utt_time_pairs, utt_words


def remove_duplicates_and_blank(hyp: List[int]) -> Tuple[List[int], List[int]]:
    # modified from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/common.py
    new_hyp: List[int] = []
    time: List[Tuple[int, int]] = []
    cur = 0
    start, end = -1, -1
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
            start = cur
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            if start != -1:
                end = cur
            cur += 1
        if start != -1 and end != -1:
            time.append((start, end))
            start, end = -1, -1
    return new_hyp, time


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    batch: dict,
    word_table: k2.SymbolTable,
    sos_id: int,
    eos_id: int,
    G: Optional[k2.Fsa] = None,
) -> Dict[str, Tuple[List[List[str]], List[List[float]]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
    - key: It indicates the setting used for decoding. For example,
           if no rescoring is used, the key is the string `no_rescore`.
           If LM rescoring is used, the key is the string `lm_scale_xxx`,
           where `xxx` is the value of `lm_scale`. An example key is
           `lm_scale_0.7`
    - value: It contains the decoding result. `len(value)` equals to
             batch size. `value[i]` is the decoding result for the i-th
             utterance in the given batch.

    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.decoding_method is "1best", it uses 1best decoding without LM rescoring.
        - params.decoding_method is "nbest", it uses nbest decoding without LM rescoring.
        - params.decoding_method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.decoding_method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph. Used only when params.decoding_method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.decoding_method is ctc-decoding.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      sos_id:
        The token ID of the SOS.
      eos_id:
        The token ID of the EOS.
      G:
        An LM. It is not None when params.decoding_method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    if HLG is not None:
        device = HLG.device
    else:
        device = H.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.simulate_streaming:
        feature_lens += params.left_context
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, params.left_context),
            value=LOG_EPS,
        )
        encoder_out, encoder_out_lens, _ = model.encoder.streaming_forward(
            x=feature,
            x_lens=feature_lens,
            chunk_size=params.decode_chunk_size,
            left_context=params.left_context,
            simulate_streaming=True,
        )
    else:
        encoder_out, encoder_out_lens = model.encoder(feature, feature_lens)

    nnet_output = model.get_ctc_output(encoder_out)
    # nnet_output is (N, T, C)

    if params.decoding_method == "ctc-greedy-search":
        timestamps, hyps = ctc_greedy_search(
            ctc_probs=nnet_output,
            nnet_output_lens=encoder_out_lens,
            sp=bpe_model,
            subsampling_factor=params.subsampling_factor,
            frame_shift_ms=params.frame_shift_ms,
        )
        key = "ctc-greedy-search"
        return {key: (hyps, timestamps)}

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"] // params.subsampling_factor,
            encoder_out_lens.cpu(),
        ),
        1,
    ).to(torch.int32)

    if H is None:
        assert HLG is not None
        decoding_graph = HLG
    else:
        assert HLG is None
        assert bpe_model is not None
        decoding_graph = H

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=decoding_graph,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    if params.decoding_method == "ctc-decoding":
        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
        timestamps, hyps = parse_fsa_timestamps_and_texts(
            best_paths=best_path,
            sp=bpe_model,
            subsampling_factor=params.subsampling_factor,
            frame_shift_ms=params.frame_shift_ms,
        )
        key = "ctc-decoding"
        return {key: (hyps, timestamps)}

    if params.decoding_method == "nbest-oracle":
        # Note: You can also pass rescored lattices to it.
        # We choose the HLG decoded lattice for speed reasons
        # as HLG decoding is faster and the oracle WER
        # is only slightly worse than that of rescored lattices.
        best_path = nbest_oracle(
            lattice=lattice,
            num_paths=params.num_paths,
            ref_texts=supervisions["text"],
            word_table=word_table,
            nbest_scale=params.nbest_scale,
            oov="<UNK>",
        )
        hyps = get_texts(best_path)
        hyps = [[word_table[i] for i in ids] for ids in hyps]
        timestamps = [[] for _ in range(len(hyps))]
        key = f"oracle_{params.num_paths}_nbest_scale_{params.nbest_scale}_hlg_scale_{params.hlg_scale}"  # noqa
        return {key: (hyps, timestamps)}

    if params.decoding_method in ["1best", "nbest"]:
        if params.decoding_method == "1best":
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
            key = f"no_rescore_hlg_scale_{params.hlg_scale}"
            timestamps, hyps = parse_fsa_timestamps_and_texts(
                best_paths=best_path,
                word_table=word_table,
                subsampling_factor=params.subsampling_factor,
                frame_shift_ms=params.frame_shift_ms,
            )
        else:
            best_path = nbest_decoding(
                lattice=lattice,
                num_paths=params.num_paths,
                use_double_scores=params.use_double_scores,
                nbest_scale=params.nbest_scale,
            )
            key = f"no_rescore-nbest-scale-{params.nbest_scale}-{params.num_paths}-hlg-scale-{params.hlg_scale}"  # noqa
            hyps = get_texts(best_path)
            hyps = [[word_table[i] for i in ids] for ids in hyps]
            timestamps = [[] for _ in range(len(hyps))]
        return {key: (hyps, timestamps)}

    assert params.decoding_method in [
        "nbest-rescoring",
        "whole-lattice-rescoring",
    ]

    lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lm_scale_list += [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lm_scale_list += [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    if params.decoding_method == "nbest-rescoring":
        best_path_dict = rescore_with_n_best_list(
            lattice=lattice,
            G=G,
            num_paths=params.num_paths,
            lm_scale_list=lm_scale_list,
            nbest_scale=params.nbest_scale,
        )
    elif params.decoding_method == "whole-lattice-rescoring":
        best_path_dict = rescore_with_whole_lattice(
            lattice=lattice,
            G_with_epsilon_loops=G,
            lm_scale_list=lm_scale_list,
        )
    else:
        assert False, f"Unsupported decoding method: {params.decoding_method}"

    ans = dict()
    if best_path_dict is not None:
        for lm_scale_str, best_path in best_path_dict.items():
            hyps = get_texts(best_path)
            hyps = [[word_table[i] for i in ids] for ids in hyps]
            timestamps = [[] for _ in range(len(hyps))]
            ans[lm_scale_str] = (hyps, timestamps)
    else:
        ans = None
    return ans


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    word_table: k2.SymbolTable,
    sos_id: int,
    eos_id: int,
    G: Optional[k2.Fsa] = None,
) -> Dict[
    str,
    List[
        Tuple[
            str,
            List[str],
            List[str],
            List[Tuple[float, float]],
            List[Tuple[float, float]],
        ]
    ],
]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph. Used only when params.decoding_method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.decoding_method is ctc-decoding.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      word_table:
        It is the word symbol table.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      G:
        An LM. It is not None when params.decoding_method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        timestamps_ref = []
        for cut in batch["supervisions"]["cut"]:
            for s in cut.supervisions:
                time = []
                if s.alignment is not None and "word" in s.alignment:
                    time = [
                        (aliword.start, aliword.end)
                        for aliword in s.alignment["word"]
                        if aliword.symbol != ""
                    ]
                timestamps_ref.append(time)

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            HLG=HLG,
            H=H,
            bpe_model=bpe_model,
            batch=batch,
            word_table=word_table,
            G=G,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        for name, (hyps, timestamps_hyp) in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts) and len(timestamps_hyp) == len(
                timestamps_ref
            )
            for cut_id, hyp_words, ref_text, time_hyp, time_ref in zip(
                cut_ids, hyps, texts, timestamps_hyp, timestamps_ref
            ):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words, time_ref, time_hyp))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[
        str,
        List[
            Tuple[
                List[str],
                List[str],
                List[str],
                List[Tuple[float, float]],
                List[Tuple[float, float]],
            ]
        ],
    ],
):
    test_set_wers = dict()
    test_set_delays = dict()
    for key, results in results_dict.items():
        recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts_and_timestamps(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w") as f:
            wer, mean_delay, var_delay = write_error_stats_with_timestamps(
                f,
                f"{test_set_name}-{key}",
                results,
                enable_log=True,
                with_end_time=True,
            )
            test_set_wers[key] = wer
            test_set_delays[key] = (mean_delay, var_delay)

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    # sort according to the mean start symbol delay
    test_set_delays = sorted(test_set_delays.items(), key=lambda x: x[1][0][0])
    delays_info = (
        params.res_dir / f"symbol-delay-summary-{test_set_name}-{params.suffix}.txt"
    )
    with open(delays_info, "w") as f:
        print("settings\t(start, end) symbol-delay (s) (start, end)", file=f)
        for key, val in test_set_delays:
            print(
                "{}\tmean: {}, variance: {}".format(key, val[0], val[1]),
                file=f,
            )

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)

    s = "\nFor {}, (start, end) symbol-delay (s) of different settings are:\n".format(
        test_set_name
    )
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_delays:
        s += "{}\tmean: {}, variance: {}{}\n".format(key, val[0], val[1], note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    assert params.decoding_method in (
        "ctc-greedy-search",
        "ctc-decoding",
        "1best",
        "nbest",
        "nbest-rescoring",
        "whole-lattice-rescoring",
        "nbest-oracle",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.simulate_streaming:
        params.suffix += f"-streaming-chunk-size-{params.decode_chunk_size}"
        params.suffix += f"-left-context-{params.left_context}"

    if params.simulate_streaming:
        assert (
            params.causal_convolution
        ), "Decoding in streaming requires causal convolution"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )
    sos_id = graph_compiler.sos_id
    eos_id = graph_compiler.eos_id

    params.vocab_size = num_classes
    params.sos_id = sos_id
    params.eos_id = eos_id

    if params.decoding_method in ["ctc-decoding", "ctc-greedy-search"]:
        HLG = None
        H = k2.ctc_topo(
            max_token=max_token_id,
            modified=False,
            device=device,
        )
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.load(str(params.lang_dir / "bpe.model"))
    else:
        H = None
        bpe_model = None
        HLG = k2.Fsa.from_dict(
            torch.load(f"{params.lang_dir}/HLG.pt", map_location=device)
        )
        assert HLG.requires_grad is False

        HLG.scores *= params.hlg_scale
        if not hasattr(HLG, "lm_scores"):
            HLG.lm_scores = HLG.scores.clone()

    if params.decoding_method in (
        "nbest-rescoring",
        "whole-lattice-rescoring",
    ):
        if not (params.lm_dir / "G_4_gram.pt").is_file():
            logging.info("Loading G_4_gram.fst.txt")
            logging.warning("It may take 8 minutes.")
            with open(params.lm_dir / "G_4_gram.fst.txt") as f:
                first_word_disambig_id = lexicon.word_table["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION: The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                # See https://github.com/k2-fsa/k2/issues/874
                # for why we need to set G.properties to None
                G.__dict__["_properties"] = None
                G = k2.Fsa.from_fsas([G]).to(device)
                G = k2.arc_sort(G)
                # Save a dummy value so that it can be loaded in C++.
                # See https://github.com/pytorch/pytorch/issues/67902
                # for why we need to do this.
                G.dummy = 1

                torch.save(G.as_dict(), params.lm_dir / "G_4_gram.pt")
        else:
            logging.info("Loading pre-compiled G_4_gram.pt")
            d = torch.load(params.lm_dir / "G_4_gram.pt", map_location=device)
            G = k2.Fsa.from_dict(d)

        if params.decoding_method == "whole-lattice-rescoring":
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)

        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        G = None

    logging.info("About to create model")
    model = get_ctc_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            HLG=HLG,
            H=H,
            bpe_model=bpe_model,
            word_table=lexicon.word_table,
            G=G,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
