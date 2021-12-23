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
from pathlib import Path
from typing import List, Tuple

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
        "--mem-dir",
        type=str,
        default="conformer_ctc/exp/mem",
        help="The experiment dir",
    )

    parser.add_argument(
        "--num-utts",
        type=int,
        default=1000,
        help="number of utts to extract memory embeddings",
    )

    parser.add_argument(
        "--mem-layer",
        type=int,
        default=None,
        help="which layer to extract memory embedding"
        "See: https://github.com/glynpu/transformers/pull/1/files",
    )

    parser.add_argument(
        "--pretrained_model",
        type=Path,
        default=None,
        help="use a pretrained model, e.g. a modle downloaded from model zoo",
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


def compute_memory(
    model: torch.nn.Module,
    processor: None,
    dl: torch.utils.data.DataLoader,
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
        memory_embeddings = model.wav2vec2(feature, mask)[0]  # [N, T, C]

        encoder_memory = memory_embeddings.to("cpu").numpy()

        cut_list = supervisions["cut"]
        assert len(cut_list) == encoder_memory.shape[0]
        assert all(c.start == 0 for c in supervisions["cut"])

        for idx, cut in enumerate(cut_list):
            num_frames = supervisions["num_samples"][idx] // 320
            cut.encoder_memory = writer.store_array(
                key=cut.id,
                value=encoder_memory[idx][:num_frames],
            )
            total_frames += num_frames

        cuts += cut_list
        logging.info(f"Processed {len(cuts)} cuts")
        if len(cuts) > params.num_utts:
            break
    return CutSet.from_cuts(cuts)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    assert args.mem_layer is not None
    assert args.mem_layer > 0 and args.mem_layer < 24

    assert args.return_cuts is True
    assert args.concatenate_cuts is False

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/mem")

    logging.info("Computing memory embedings- started")
    logging.info(params)

    logging.info("About to create model")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self",
        output_layer_index=params.mem_layer,
    ).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self"
    )
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    params["device"] = device

    model.to(device)
    model.eval()

    librispeech = LibriSpeechAsrDataModule(args)

    test_dl = librispeech.test_dataloaders()  # a list

    mem_dir = Path(params.mem_dir)
    mem_dir.mkdir(exist_ok=True)

    enabled_datasets = {
        "test_clean": test_dl[0],
    }

    with NumpyHdf5Writer(
        mem_dir / f"{args.mem_layer}layer-memory_embeddings"
    ) as writer:
        for name, dl in enabled_datasets.items():
            cut_set = compute_memory(
                model=model,
                processor=processor,
                dl=dl,
                params=params,
                writer=writer,
            )
            cut_set.to_json(
                mem_dir / f"{args.mem_layer}layer-memory_manifest.json.gz"
            )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
