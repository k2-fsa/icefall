#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
from pathlib import Path
from typing import List, Tuple

import k2
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.decode import one_best_decoding
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    encode_supervisions,
    get_alignments,
    save_alignments,
    setup_logger,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=34,
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
        "--lang-dir",
        type=str,
        default="data/lang_bpe",
        help="The lang dir",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--ali-dir",
        type=str,
        default="data/ali_500",
        help="The experiment dir",
    )
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "lm_dir": Path("data/lm"),
            "feature_dim": 80,
            "nhead": 8,
            "attention_dim": 512,
            "subsampling_factor": 4,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "output_beam": 10,
            "use_double_scores": True,
        }
    )
    return params


def compute_alignments(
    model: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
) -> List[Tuple[str, List[int]]]:
    """Compute the framewise alignments of a dataset.

    Args:
      model:
        The neural network model.
      dl:
        Dataloader containing the dataset.
      params:
        Parameters for computing alignments.
      graph_compiler:
        It converts token IDs to decoding graphs.
    Returns:
      Return a list of tuples. Each tuple contains two entries:
        - Utterance ID
        - Framewise alignments (token IDs) after subsampling
    """
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
    num_cuts = 0

    device = graph_compiler.device
    ans = []
    for batch_idx, batch in enumerate(dl):
        feature = batch["inputs"]

        # at entry, feature is [N, T, C]
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]

        cut_ids = []
        for cut in supervisions["cut"]:
            assert len(cut.supervisions) == 1
            cut_ids.append(cut.id)

        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        # nnet_output is [N, T, C]
        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=params.subsampling_factor
        )
        # we need also to sort cut_ids as encode_supervisions()
        # reorders "texts".
        # In general, new2old is an identity map since lhotse sorts the returned
        # cuts by duration in descending order
        new2old = supervision_segments[:, 0].tolist()
        cut_ids = [cut_ids[i] for i in new2old]

        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)

        dense_fsa_vec = k2.DenseFsaVec(
            nnet_output,
            supervision_segments,
            allow_truncate=params.subsampling_factor - 1,
        )

        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            params.output_beam,
        )

        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=params.use_double_scores,
        )

        ali_ids = get_alignments(best_path)
        assert len(ali_ids) == len(cut_ids)
        ans += list(zip(cut_ids, ali_ids))

        num_cuts += len(ali_ids)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(
                f"batch {batch_str}, cuts processed until now is {num_cuts}"
            )

    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    assert args.return_cuts is True
    assert args.concatenate_cuts is False
    if args.full_libri is False:
        print("Changing --full-libri to True")
        args.full_libri = True

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/ali")

    logging.info("Computing alignment - started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
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
        model.load_state_dict(average_checkpoints(filenames))

    model.to(device)
    model.eval()

    librispeech = LibriSpeechAsrDataModule(args)

    train_dl = librispeech.train_dataloaders()
    valid_dl = librispeech.valid_dataloaders()
    test_dl = librispeech.test_dataloaders()  # a list

    ali_dir = Path(params.ali_dir)
    ali_dir.mkdir(exist_ok=True)

    enabled_datasets = {
        "test_clean": test_dl[0],
        "test_other": test_dl[1],
        "train-960": train_dl,
        "valid": valid_dl,
    }
    # For train-960, it takes about 3 hours 40 minutes, i.e., 3.67 hours to
    # compute the alignments if you use --max-duration=500
    #
    # There are 960 * 3 = 2880 hours data and it takes only
    # 3 hours 40 minutes to get the alignment.
    # The RTF is roughly: 3.67 / 2880 = 0.0012743
    for name, dl in enabled_datasets.items():
        logging.info(f"Processing {name}")
        if name == "train-960":
            logging.info(
                "It will take about 3 hours 40 minutes for {name}, "
                "which contains 960 * 3 = 2880 hours of data"
            )
        alignments = compute_alignments(
            model=model,
            dl=dl,
            params=params,
            graph_compiler=graph_compiler,
        )
        num_utt = len(alignments)
        alignments = dict(alignments)
        assert num_utt == len(alignments)
        filename = ali_dir / f"{name}.pt"
        save_alignments(
            alignments=alignments,
            subsampling_factor=params.subsampling_factor,
            filename=filename,
        )
        logging.info(
            f"For dataset {name}, its alignments are saved to {filename}"
        )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
