#!/usr/bin/env python3


import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import YesNoAsrDataModule
from model import Tdnn

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.decode import get_lattice, one_best_decoding
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=14,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=2,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--export",
        type=str2bool,
        default=False,
        help="""When enabled, the averaged model is saved to
        tdnn/exp/pretrained.pt. Note: only model.state_dict() is saved.
        pretrained.pt contains a dict {"model": model.state_dict()},
        which can be loaded by `icefall.checkpoint.load_checkpoint()`.
        """,
    )
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "exp_dir": Path("tdnn/exp/"),
            "lang_dir": Path("data/lang_phone"),
            "lm_dir": Path("data/lm"),
            "feature_dim": 23,
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
    HLG: k2.Fsa,
    batch: dict,
    word_table: k2.SymbolTable,
) -> List[List[int]]:
    """Decode one batch and return the result in a list-of-list.
    Each sub list contains the word IDs for an utterance in the batch.

    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding.
        - params.method is "nbest", it uses nbest decoding.

      model:
        The neural model.
      HLG:
        The decoding graph.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
        (https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py)
      word_table:
        It is the word symbol table.
    Returns:
      Return the decoding result. `len(ans)` == batch size.
    """
    device = HLG.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    nnet_output = model(feature)
    # nnet_output is (N, T, C)

    batch_size = nnet_output.shape[0]
    supervision_segments = torch.tensor(
        [[i, 0, nnet_output.shape[1]] for i in range(batch_size)],
        dtype=torch.int32,
    )

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
    )

    best_path = one_best_decoding(
        lattice=lattice, use_double_scores=params.use_double_scores
    )
    hyps = get_texts(best_path)
    hyps = [[word_table[i] for i in ids] for ids in hyps]
    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: k2.Fsa,
    word_table: k2.SymbolTable,
) -> List[Tuple[str, List[str], List[str]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph.
      word_table:
        It is word symbol table.
    Returns:
      Return a tuple contains two elements (ref_text, hyp_text):
      The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = []
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps = decode_one_batch(
            params=params,
            model=model,
            HLG=HLG,
            batch=batch,
            word_table=word_table,
        )

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_words = ref_text.split()
            this_batch.append((cut_id, ref_words, hyp_words))

        results.extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    exp_dir: Path,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
) -> None:
    """Save results to `exp_dir`.
    Args:
      exp_dir:
        The output directory. This function create the following files inside
        this directory:

            - recogs-{test_set_name}.text

                It contains the reference and hypothesis results, like below::

                    ref=['NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'YES']
                    hyp=['NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'YES']
                    ref=['NO', 'NO', 'YES', 'NO', 'YES', 'NO', 'NO', 'YES']
                    hyp=['NO', 'NO', 'YES', 'NO', 'YES', 'NO', 'NO', 'YES']

            - errs-{test_set_name}.txt

                It contains the detailed WER.
      test_set_name:
        The name of the test set, which will be part of the result filename.
      results:
        A list of tuples, each of which contains (ref_words, hyp_words).
    Returns:
      Return None.
    """
    recog_path = exp_dir / f"recogs-{test_set_name}.txt"
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = exp_dir / f"errs-{test_set_name}.txt"
    with open(errs_filename, "w") as f:
        write_error_stats(f, f"{test_set_name}", results)

    logging.info("Wrote detailed error stats to {}".format(errs_filename))


@torch.no_grad()
def main():
    parser = get_parser()
    YesNoAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))
    params["env_info"] = get_env_info()

    setup_logger(f"{params.exp_dir}/log/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    HLG = k2.Fsa.from_dict(torch.load(f"{params.lang_dir}/HLG.pt", map_location="cpu"))
    HLG = HLG.to(device)
    assert HLG.requires_grad is False

    model = Tdnn(
        num_features=params.feature_dim,
        num_classes=max_token_id + 1,  # +1 for the blank symbol
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
        model.load_state_dict(average_checkpoints(filenames))

    if params.export:
        logging.info(f"Export averaged model to {params.exp_dir}/pretrained.pt")
        torch.save({"model": model.state_dict()}, f"{params.exp_dir}/pretrained.pt")
        return

    model.to(device)
    model.eval()

    # we need cut ids to display recognition results.
    args.return_cuts = True
    yes_no = YesNoAsrDataModule(args)
    test_dl = yes_no.test_dataloaders()
    results = decode_dataset(
        dl=test_dl,
        params=params,
        model=model,
        HLG=HLG,
        word_table=lexicon.word_table,
    )

    save_results(exp_dir=params.exp_dir, test_set_name="test_set", results=results)

    logging.info("Done!")


if __name__ == "__main__":
    main()
