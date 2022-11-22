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

"""
Usage:
    ./conformer_ctc/ali.py \
            --exp-dir ./conformer_ctc/exp \
            --lang-dir ./data/lang_bpe_500 \
            --epoch 20 \
            --avg 10 \
            --max-duration 300 \
            --dataset train-clean-100 \
            --out-dir data/ali
"""

import argparse
import logging
from pathlib import Path

import k2
import numpy as np
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from lhotse import CutSet
from lhotse.features.io import FeaturesWriter, NumpyHdf5Writer

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.decode import one_best_decoding
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    encode_supervisions,
    get_alignments,
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
        default="data/lang_bpe_500",
        help="The lang dir",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="""Output directory.
        It contains 3 generated files:

        - labels_xxx.h5
        - aux_labels_xxx.h5
        - cuts_xxx.json.gz

        where xxx is the value of `--dataset`. For instance, if
        `--dataset` is `train-clean-100`, it will contain 3 files:

        - `labels_train-clean-100.h5`
        - `aux_labels_train-clean-100.h5`
        - `cuts_train-clean-100.json.gz`

        Note: Both labels_xxx.h5 and aux_labels_xxx.h5 contain framewise
        alignment. The difference is that labels_xxx.h5 contains repeats.
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""The name of the dataset to compute alignments for.
        Possible values are:
            - test-clean.
            - test-other
            - train-clean-100
            - train-clean-360
            - train-other-500
            - dev-clean
            - dev-other
        """,
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
            # Set it to 0 since attention decoder
            # is not used for computing alignments
            "num_decoder_layers": 0,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "output_beam": 10,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params


def compute_alignments(
    model: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    labels_writer: FeaturesWriter,
    aux_labels_writer: FeaturesWriter,
    params: AttributeDict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
) -> CutSet:
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
      Return a CutSet. Each cut has two custom fields: labels_alignment
      and aux_labels_alignment, containing framewise alignments information.
      Both are of type `lhotse.array.TemporalArray`. The difference between
      the two alignments is that `labels_alignment` contain repeats.
    """
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
    num_cuts = 0

    device = graph_compiler.device
    cuts = []
    for batch_idx, batch in enumerate(dl):
        feature = batch["inputs"]

        # at entry, feature is [N, T, C]
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        cut_list = supervisions["cut"]

        for cut in cut_list:
            assert len(cut.supervisions) == 1, f"{len(cut.supervisions)}"

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

        cut_list = [cut_list[i] for i in new2old]

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

        labels_ali = get_alignments(best_path, kind="labels")
        aux_labels_ali = get_alignments(best_path, kind="aux_labels")
        assert len(labels_ali) == len(aux_labels_ali) == len(cut_list)
        for cut, labels, aux_labels in zip(cut_list, labels_ali, aux_labels_ali):
            cut.labels_alignment = labels_writer.store_array(
                key=cut.id,
                value=np.asarray(labels, dtype=np.int32),
                # frame shift is 0.01s, subsampling_factor is 4
                frame_shift=0.04,
                temporal_dim=0,
                start=0,
            )
            cut.aux_labels_alignment = aux_labels_writer.store_array(
                key=cut.id,
                value=np.asarray(aux_labels, dtype=np.int32),
                # frame shift is 0.01s, subsampling_factor is 4
                frame_shift=0.04,
                temporal_dim=0,
                start=0,
            )

        cuts += cut_list

        num_cuts += len(cut_list)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return CutSet.from_cuts(cuts)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    args.enable_spec_aug = False
    args.enable_musan = False
    args.return_cuts = True
    args.concatenate_cuts = False

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-ali")

    logging.info(f"Computing alignments for {params.dataset} - started")
    logging.info(params)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(exist_ok=True)

    out_labels_ali_filename = out_dir / f"labels_{params.dataset}.h5"
    out_aux_labels_ali_filename = out_dir / f"aux_labels_{params.dataset}.h5"
    out_manifest_filename = out_dir / f"cuts_{params.dataset}.json.gz"

    for f in (
        out_labels_ali_filename,
        out_aux_labels_ali_filename,
        out_manifest_filename,
    ):
        if f.exists():
            logging.info(f"{f} exists - skipping")
            return

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

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
    model.to(device)

    if params.avg == 1:
        load_checkpoint(
            f"{params.exp_dir}/epoch-{params.epoch}.pt", model, strict=False
        )
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.eval()

    librispeech = LibriSpeechAsrDataModule(args)
    if params.dataset == "test-clean":
        test_clean_cuts = librispeech.test_clean_cuts()
        dl = librispeech.test_dataloaders(test_clean_cuts)
    elif params.dataset == "test-other":
        test_other_cuts = librispeech.test_other_cuts()
        dl = librispeech.test_dataloaders(test_other_cuts)
    elif params.dataset == "train-clean-100":
        train_clean_100_cuts = librispeech.train_clean_100_cuts()
        dl = librispeech.train_dataloaders(train_clean_100_cuts)
    elif params.dataset == "train-clean-360":
        train_clean_360_cuts = librispeech.train_clean_360_cuts()
        dl = librispeech.train_dataloaders(train_clean_360_cuts)
    elif params.dataset == "train-other-500":
        train_other_500_cuts = librispeech.train_other_500_cuts()
        dl = librispeech.train_dataloaders(train_other_500_cuts)
    elif params.dataset == "dev-clean":
        dev_clean_cuts = librispeech.dev_clean_cuts()
        dl = librispeech.valid_dataloaders(dev_clean_cuts)
    else:
        assert params.dataset == "dev-other", f"{params.dataset}"
        dev_other_cuts = librispeech.dev_other_cuts()
        dl = librispeech.valid_dataloaders(dev_other_cuts)

    logging.info(f"Processing {params.dataset}")
    with NumpyHdf5Writer(out_labels_ali_filename) as labels_writer:
        with NumpyHdf5Writer(out_aux_labels_ali_filename) as aux_labels_writer:
            cut_set = compute_alignments(
                model=model,
                dl=dl,
                labels_writer=labels_writer,
                aux_labels_writer=aux_labels_writer,
                params=params,
                graph_compiler=graph_compiler,
            )

    cut_set.to_file(out_manifest_filename)

    logging.info(
        f"For dataset {params.dataset}, its alignments with repeats are "
        f"saved to {out_labels_ali_filename}, the alignments without repeats "
        f"are saved to {out_aux_labels_ali_filename}, and the cut manifest "
        f"file is {out_manifest_filename}. Number of cuts: {len(cut_set)}"
    )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
