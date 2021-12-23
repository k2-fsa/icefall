#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (author: Liyong Guo)
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
import os
from pathlib import Path
from typing import List, Tuple
from quantization import Quantizer

import numpy as np
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse.features.io import NumpyHdf5Writer
from lhotse import CutSet

from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


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
        default=1,
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
        "--data-dir",
        type=Path,
        default="./data/",
        help="The experiment dir",
    )

    parser.add_argument(
        "--mem-dir",
        type=Path,
        default="conformer_ctc/exp/mem",
        help="The experiment dir",
    )

    parser.add_argument(
        "--quantizer-id",
        type=str,
        default=None,
        help="quantizer_id" "Manully set this incase of mistake.",
    )

    parser.add_argument(
        "--bytes-per-frame",
        type=int,
        default=4,
        help="The number of bytes to use to quantize each memory embeddings",
    )

    parser.add_argument(
        "--memory-embedding-dim",
        type=int,
        default=512,
        help="dim of memory embeddings to train quantizer",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="which subset to extract codebook index"
        "clean-100, clean-360, other-500",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="wav2vec",
        help="a short str to introduce which models the embeddings come from",
    )

    parser.add_argument(
        "--mem-layer",
        type=int,
        default=None,
        help="which layer to extract memory embedding"
        "Set this manully incase of mistake.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 80,
            "nhead": 8,
            "attention_dim": 512,
            "subsampling_factor": 4,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "output_beam": 10,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params


def compute_codeindices(
    model: torch.nn.Module,
    processor: None,
    dl: torch.utils.data.DataLoader,
    quantizer: None,
    params: AttributeDict,
    writer: None,
) -> List[Tuple[str, List[int]]]:
    """Compute the framewise alignments of a dataset.

    Args:
      model:
        The neural network model.
      dl:
        Dataloader containing the dataset.
      params:
        Parameters for computing memory.
    Returns:
      Return a list of tuples. Each tuple contains two entries:
        - Utterance ID
        - memory embeddings
    """
    num_cuts = 0

    cuts = []
    total_frames = 0
    for batch_idx, batch in enumerate(dl):
        inputs = processor(
            batch["inputs"],
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
        )
        feature = inputs["input_values"].squeeze(0)
        feature = feature.to(model.device)
        B, T = feature.shape

        supervisions = batch["supervisions"]
        num_samples = supervisions["num_samples"]
        mask = torch.arange(0, T).expand(B, T) < num_samples.reshape([-1, 1])
        mask = mask.to(model.device)
        encoder_memory = model.wav2vec2(feature, mask)[0]  # [N, T, C]

        codebook_indices = quantizer.encode(encoder_memory)

        # [N, T, C]
        codebook_indices = codebook_indices.to("cpu").numpy().astype(np.int16)

        cut_list = supervisions["cut"]
        assert len(cut_list) == codebook_indices.shape[0]

        assert all(c.start == 0 for c in supervisions["cut"])
        for idx, cut in enumerate(cut_list):
            num_frames = supervisions["num_samples"][idx] // 320
            cut.codebook_indices = writer.store_array(
                key=cut.id,
                value=codebook_indices[idx][:num_frames],
                frame_shift=0.02,
                temporal_dim=0,
                start=0,
            )
            total_frames += num_frames

        cuts += cut_list
        num_cuts += len(cut_list)
        logging.info(
            f"processed {total_frames} frames and {num_cuts} cuts;"
            "{batch_idx} of {num_batches}"
        )
    return CutSet.from_cuts(cuts)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    assert args.subset in ["clean-100", "clean-360", "other-500"], args.subset
    # disable augmentation when extracting codebook index
    assert args.enable_augmentation is False

    # Manully set options
    assert args.quantizer_id is not None
    assert args.model_id is not None
    assert args.mem_layer is not None

    assert args.return_cuts is True
    assert args.concatenate_cuts is False

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/codebook_index")

    logging.info("Computing memory embedings started")
    logging.info(params)

    logging.info("About to create model")
    quantizer_fn = (
        params.mem_dir
        / f"{params.mem_layer}layer-{params.quantizer_id}-bytes_per_frame_{params.bytes_per_frame}-quantizer.pt"  # noqa: E501
    )
    assert os.path.isfile(quantizer_fn), f"{quantizer_fn}"
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self",
        mem_layer=params.mem_layer,
    ).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self"
    )

    quantizer = Quantizer(
        dim=params.memory_embedding_dim,
        num_codebooks=args.bytes_per_frame,
        codebook_size=256,
    )
    quantizer.load_state_dict(torch.load(quantizer_fn))
    quantizer = quantizer.to("cuda")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    params["device"] = device

    model.to(device)
    model.eval()

    librispeech = LibriSpeechAsrDataModule(args)

    train_dl = librispeech.train_dataloaders()

    cdidx_dir = (
        Path(params.data_dir)
        / f"{args.model_id}-{args.mem_layer}layer-{args.quantizer_id}-bytes_per_frame-{args.bytes_per_frame}"  # noqa: E501
    )
    cdidx_dir.mkdir(exist_ok=True)

    with NumpyHdf5Writer(
        cdidx_dir
        / f"{args.model_id}-{args.mem_layer}layer-cdidx_train-{args.subset}"
    ) as writer:
        cut_set = compute_codeindices(
            model=model,
            processor=processor,
            dl=train_dl,
            quantizer=quantizer,
            params=params,
            writer=writer,
        )
        cut_set.to_json(cdidx_dir / f"cuts_train-{args.subset}.json.gz")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
