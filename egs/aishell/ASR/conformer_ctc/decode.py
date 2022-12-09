#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo,
#                                            Fangjun Kuang,
#                                            Wei Kang)
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

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule
from conformer import Conformer

from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder,
)
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=49,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=20,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="attention-decoder",
        help="""Decoding method.
        Supported values are:
            - (0) ctc-decoding. Use CTC decoding. It maps the tokens ids to
              tokens using token symbol tabel directly.
            - (1) 1best. Extract the best path from the decoding lattice as the
              decoding result.
            - (2) nbest. Extract n paths from the decoding lattice; the path
              with the highest score is the decoding result.
            - (3) attention-decoder. Extract n paths from the lattice,
              the path with the highest score is the decoding result.
            - (4) nbest-oracle. Its WER is the lower bound of any n-best
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
        nbest, attention-decoder, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, attention-decoder, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_char",
        help="The lang dir",
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="""The LM dir.
        It should contain either G_3_gram.pt or G_3_gram.fst.txt
        """,
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "subsampling_factor": 4,
            "feature_dim": 80,
            "nhead": 4,
            "attention_dim": 512,
            "num_encoder_layers": 12,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            # parameters for decoder
            "search_beam": 20,
            "output_beam": 7,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    batch: dict,
    lexicon: Lexicon,
    sos_id: int,
    eos_id: int,
) -> Dict[str, List[List[int]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if decoding method is 1best, the key is the string `no_rescore`.
               If attention rescoring is used, the key is the string
               `ngram_lm_scale_xxx_attention_scale_xxx`, where `xxx` is the
               value of `lm_scale` and `attention_scale`. An example key is
               `ngram_lm_scale_0.7_attention_scale_0.5`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "attention-decoder", it uses attention rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph. Used when params.method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      lexicon:
        It contains the token symbol table and the word symbol table.
      sos_id:
        The token ID of the SOS.
      eos_id:
        The token ID of the EOS.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
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

    nnet_output, memory, memory_key_padding_mask = model(feature, supervisions)
    # nnet_output is (N, T, C)

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"] // params.subsampling_factor,
            supervisions["num_frames"] // params.subsampling_factor,
        ),
        1,
    ).to(torch.int32)

    if H is None:
        assert HLG is not None
        decoding_graph = HLG
    else:
        assert HLG is None
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

    if params.method == "ctc-decoding":
        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
        # Note: `best_path.aux_labels` contains token IDs, not word IDs
        # since we are using H, not HLG here.
        #
        # token_ids is a lit-of-list of IDs
        token_ids = get_texts(best_path)

        key = "ctc-decoding"
        hyps = [[lexicon.token_table[i] for i in ids] for ids in token_ids]
        return {key: hyps}

    if params.method == "nbest-oracle":
        # Note: You can also pass rescored lattices to it.
        # We choose the HLG decoded lattice for speed reasons
        # as HLG decoding is faster and the oracle WER
        # is only slightly worse than that of rescored lattices.
        best_path = nbest_oracle(
            lattice=lattice,
            num_paths=params.num_paths,
            ref_texts=supervisions["text"],
            word_table=lexicon.word_table,
            nbest_scale=params.nbest_scale,
            oov="<UNK>",
        )
        hyps = get_texts(best_path)
        hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
        key = f"oracle_{params.num_paths}_nbest_scale_{params.nbest_scale}"  # noqa
        return {key: hyps}

    if params.method in ["1best", "nbest"]:
        if params.method == "1best":
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
            key = f"no_rescore-scale-{params.nbest_scale}-{params.num_paths}"  # noqa

        hyps = get_texts(best_path)
        hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
        return {key: hyps}

    assert params.method == "attention-decoder"

    best_path_dict = rescore_with_attention_decoder(
        lattice=lattice,
        num_paths=params.num_paths,
        model=model,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
        sos_id=sos_id,
        eos_id=eos_id,
        nbest_scale=params.nbest_scale,
    )
    ans = dict()
    if best_path_dict is not None:
        for lm_scale_str, best_path in best_path_dict.items():
            hyps = get_texts(best_path)
            hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
            ans[lm_scale_str] = hyps
    return ans


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    lexicon: Lexicon,
    sos_id: int,
    eos_id: int,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph. Used when params.method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.method is ctc-decoding.
      lexicon:
        It contains the token symbol table and the word symbol table.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
    Returns:
      Return a dict, whose key may be "no-rescore" if the decoding method is
      1best or it may be "ngram_lm_scale_0.7_attention_scale_0.5" if attention
      rescoring is used. Its value is a list of tuples. Each tuple contains two
      elements: The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

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
            HLG=HLG,
            H=H,
            batch=batch,
            lexicon=lexicon,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    if params.method == "attention-decoder":
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.exp_dir / f"recogs-{test_set_name}-{key}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.exp_dir / f"errs-{test_set_name}-{key}.txt"
        # we compute CER for aishell dataset.
        results_char = []
        for res in results:
            results_char.append((res[0], list("".join(res[1])), list("".join(res[2]))))
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results_char, enable_log=enable_log
            )
            test_set_wers[key] = wer

        if enable_log:
            logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"cer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("settings\tCER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, CER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-{params.method}/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )
    sos_id = graph_compiler.sos_id
    eos_id = graph_compiler.eos_id

    if params.method == "ctc-decoding":
        HLG = None
        H = k2.ctc_topo(
            max_token=max_token_id,
            modified=False,
            device=device,
        )
    else:
        H = None
        HLG = k2.Fsa.from_dict(
            torch.load(f"{params.lang_dir}/HLG.pt", map_location=device)
        )
        assert HLG.requires_grad is False

        if not hasattr(HLG, "lm_scores"):
            HLG.lm_scores = HLG.scores.clone()

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    aishell = AishellAsrDataModule(args)
    test_cuts = aishell.test_cuts()
    test_dl = aishell.test_dataloaders(test_cuts)

    test_sets = ["test"]
    test_dls = [test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            HLG=HLG,
            H=H,
            lexicon=lexicon,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
