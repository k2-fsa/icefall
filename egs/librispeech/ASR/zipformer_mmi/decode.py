#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Liyong Guo,
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
(1) 1best
./zipformer_mmi/mmi_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer_mmi/exp \
    --max-duration 100 \
    --decoding-method 1best
(2) nbest
./zipformer_mmi/mmi_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer_mmi/exp \
    --max-duration 100 \
    --nbest-scale 1.0 \
    --decoding-method nbest
(3) nbest-rescoring-LG
./zipformer_mmi/mmi_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer_mmi/exp \
    --max-duration 100 \
    --nbest-scale 1.0 \
    --decoding-method nbest-rescoring-LG
(4) nbest-rescoring-3-gram
./zipformer_mmi/mmi_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer_mmi/exp \
    --max-duration 100 \
    --nbest-scale 1.0 \
    --decoding-method nbest-rescoring-3-gram
(5) nbest-rescoring-4-gram
./zipformer_mmi/mmi_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer_mmi/exp \
    --max-duration 100 \
    --nbest-scale 1.0 \
    --decoding-method nbest-rescoring-4-gram
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

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_rescore_with_LM,
    one_best_decoding,
)
from icefall.lexicon import Lexicon
from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
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
        default="zipformer_mmi/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
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
        default="1best",
        help="""Decoding method. Use HP as decoding graph, where H is
        ctc_topo and P is token-level bi-gram lm.
        Supported values are:
        - (1) 1best. Extract the best path from the decoding lattice as the
          decoding result.
        - (2) nbest. Extract n paths from the decoding lattice; the path
          with the highest score is the decoding result.
        - (4) nbest-rescoring-LG. Extract n paths from the decoding lattice,
          rescore them with an word-level 3-gram LM, the path with the
          highest score is the decoding result.
        - (5) nbest-rescoring-3-gram. Extract n paths from the decoding
          lattice, rescore them with an token-level 3-gram LM, the path with
          the highest score is the decoding result.
        - (6) nbest-rescoring-4-gram. Extract n paths from the decoding
          lattice, rescore them with an token-level 4-gram LM, the path with
          the highest score is the decoding result.
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
        default=1.0,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--hp-scale",
        type=float,
        default=1.0,
        help="""The scale to be applied to `ctc_topo_P.scores`.
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


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    HP: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    batch: dict,
    G: Optional[k2.Fsa] = None,
    LG: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[str]]]:
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
        - params.decoding_method is "nbest-rescoring-LG", it uses nbest rescoring with word-level 3-gram LM.
        - params.decoding_method is "nbest-rescoring-3-gram", it uses nbest rescoring with token-level 3-gram LM.
        - params.decoding_method is "nbest-rescoring-4-gram", it uses nbest rescoring with token-level 4-gram LM.

      model:
        The neural model.
      HP:
        The decoding graph. H is ctc_topo, P is token-level bi-gram LM.
      bpe_model:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      LG:
        An LM. L is the lexicon, G is a word-level 3-gram LM.
        It is used when params.decoding_method is "nbest-rescoring-LG".
      G:
        An LM. L is the lexicon, G is a token-level 3-gram or 4-gram LM.
        It is used when params.decoding_method is "nbest-rescoring-3-gram"
        or "nbest-rescoring-4-gram".
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    device = HP.device
    feature = batch["inputs"]
    assert feature.ndim == 3, feature.shape
    feature = feature.to(device)

    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    nnet_output, encoder_out_lens = model(x=feature, x_lens=feature_lens)
    # nnet_output is (N, T, C)

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"] // params.subsampling_factor,
            supervisions["num_frames"] // params.subsampling_factor,
        ),
        1,
    ).to(torch.int32)

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=HP,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    method = params.decoding_method

    if method in ["1best", "nbest"]:
        if method == "1best":
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
            key = "no_rescore"
        else:
            best_path = nbest_decoding(
                lattice=lattice,
                num_paths=params.num_paths,
                use_double_scores=params.use_double_scores,
                nbest_scale=params.nbest_scale,
            )
            key = f"no_rescore-nbest-scale-{params.nbest_scale}-{params.num_paths}"  # noqa

        # Note: `best_path.aux_labels` contains token IDs, not word IDs
        # since we are using HP, not HLG here.
        #
        # token_ids is a lit-of-list of IDs
        token_ids = get_texts(best_path)
        # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
        hyps = bpe_model.decode(token_ids)
        # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
        hyps = [s.split() for s in hyps]
        return {key: hyps}

    assert method in [
        "nbest-rescoring-LG",  # word-level 3-gram lm
        "nbest-rescoring-3-gram",  # token-level 3-gram lm
        "nbest-rescoring-4-gram",  # token-level 4-gram lm
    ]

    lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lm_scale_list += [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lm_scale_list += [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    if method == "nbest-rescoring-LG":
        assert LG is not None
        LM = LG
    else:
        assert G is not None
        LM = G
    best_path_dict = nbest_rescore_with_LM(
        lattice=lattice,
        LM=LM,
        num_paths=params.num_paths,
        lm_scale_list=lm_scale_list,
        nbest_scale=params.nbest_scale,
    )

    ans = dict()
    suffix = f"-nbest-scale-{params.nbest_scale}-{params.num_paths}"
    for lm_scale_str, best_path in best_path_dict.items():
        token_ids = get_texts(best_path)
        # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
        hyps = bpe_model.decode(token_ids)
        # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
        hyps = [s.split() for s in hyps]
        ans[lm_scale_str + suffix] = hyps
    return ans


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HP: k2.Fsa,
    bpe_model: spm.SentencePieceProcessor,
    G: Optional[k2.Fsa] = None,
    LG: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HP:
        The decoding graph. H is ctc_topo, P is token-level bi-gram LM.
      bpe_model:
        The BPE model.
      LG:
        An LM. L is the lexicon, G is a word-level 3-gram LM.
        It is used when params.decoding_method is "nbest-rescoring-LG".
      G:
        An LM. L is the lexicon, G is a token-level 3-gram or 4-gram LM.
        It is used when params.decoding_method is "nbest-rescoring-3-gram"
        or "nbest-rescoring-4-gram".

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

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            HP=HP,
            bpe_model=bpe_model,
            batch=batch,
            G=G,
            LG=LG,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(f, f"{test_set_name}-{key}", results)
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    assert params.decoding_method in (
        "1best",
        "nbest",
        "nbest-rescoring-LG",  # word-level 3-gram lm
        "nbest-rescoring-3-gram",  # token-level 3-gram lm
        "nbest-rescoring-4-gram",  # token-level 4-gram lm
    ), params.decoding_method
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    params.vocab_size = num_classes
    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = 0

    bpe_model = spm.SentencePieceProcessor()
    bpe_model.load(str(params.lang_dir / "bpe.model"))
    mmi_graph_compiler = MmiTrainingGraphCompiler(
        params.lang_dir,
        uniq_filename="lexicon.txt",
        device=device,
        oov="<UNK>",
        sos_id=1,
        eos_id=1,
    )
    HP = mmi_graph_compiler.ctc_topo_P
    HP.scores *= params.hp_scale
    if not hasattr(HP, "lm_scores"):
        HP.lm_scores = HP.scores.clone()

    LG = None
    G = None

    if params.decoding_method == "nbest-rescoring-LG":
        lg_filename = params.lang_dir / "LG.pt"
        logging.info(f"Loading {lg_filename}")
        LG = k2.Fsa.from_dict(torch.load(lg_filename, map_location=device))
        LG = k2.Fsa.from_fsas([LG]).to(device)
        LG.lm_scores = LG.scores.clone()

    elif params.decoding_method in ["nbest-rescoring-3-gram", "nbest-rescoring-4-gram"]:
        order = params.decoding_method[-6]
        assert order in ("3", "4"), (params.decoding_method, order)
        order = int(order)
        if not (params.lang_dir / f"{order}gram.pt").is_file():
            logging.info(f"Loading {order}gram.fst.txt")
            logging.warning("It may take a few minutes.")
            with open(params.lang_dir / f"{order}gram.fst.txt") as f:
                first_token_disambig_id = lexicon.token_table["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION: The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_token_disambig_id] = 0
                G = k2.Fsa.from_fsas([G]).to(device)
                # G = k2.remove_epsilon(G)
                G = k2.arc_sort(G)
                # Save a dummy value so that it can be loaded in C++.
                # See https://github.com/pytorch/pytorch/issues/67902
                # for why we need to do this.
                G.dummy = 1

                torch.save(G.as_dict(), params.lang_dir / f"{order}gram.pt")
        else:
            logging.info(f"Loading pre-compiled {order}gram.pt")
            d = torch.load(params.lang_dir / f"{order}gram.pt", map_location=device)
            G = k2.Fsa.from_dict(d)

        G.lm_scores = G.scores.clone()

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
            HP=HP,
            bpe_model=bpe_model,
            G=G,
            LG=LG,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
