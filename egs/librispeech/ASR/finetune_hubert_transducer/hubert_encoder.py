#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
import logging
from typing import Tuple

import fairseq
import torch
from encoder_interface import EncoderInterface

from icefall.utils import make_pad_mask, str2bool


class HubertEncoder(EncoderInterface):
    def __init__(
        self,
        model_dir: str,
        output_size: int = 768,
        freeze_finetune_updates: int = 8000,
        mask_prob: float = 0.65,
        mask_channel_prob: float = 0.5,
        mask_channel_length: int = 64,
        subsample_output: bool = False,
        subsample_mode: str = None,
        training: bool = True,
    ):
        """Definition of a Hubert encoder

        Args:
            model_dir (str):
                Path to the pretrained Hubert model issued by fairseq. Please
                refer to `download.sh` for instructions.
            output_size (int, optional):
                The output dimension of HubertEncoder.
            freeze_finetune_updates (int, optional):
                The number of steps during which Hubert model frozen.
            mask_prob (float, optional):
                The probability of applying mask.
                Please refer to (https://arxiv.org/abs/2006.11477)
            mask_channel_prob (float, optional):
                The probability of masking channel. Same reference as above.
            mask_channel_length (int, optional):
                The length of channels to be masked. Same reference as above.
            subsample_output (bool, optional):
                Apply subsample to HubertEncoder. Hubert model has
                an encoder rate of 50Hz, which is twice of most E2E
                ASR models. Set this to true apply subsampling to reduce
                encoder rate to 25Hz.
            subsample_mode (str, optional):
                Which type of subsampling module to be used. If
                `subsample_output=false` or `subsample_mode=None`, no
                subsample module is applied. Supported subsample module
                includes ['concat_tanh', 'concat_relu', 'concat']. All these
                methods concatenate each two successive frames and then
                apply a linear layer (with different activation functions)

            training (bool, optional): The status of training.
        """

        super().__init__()
        (
            models,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_dir])
        model = models[0]

        self.encoders = model
        self.hubert_output_size = cfg["model"]["encoder_embed_dim"]
        if mask_channel_length != model.mask_channel_length:
            logging.warning(
                "Overwriting mask channel length to {}".format(
                    mask_channel_length
                )
            )
            model.mask_channel_length = mask_channel_length
        if mask_channel_prob != model.mask_channel_prob:
            logging.warning(
                "Overwriting mask channel prob to {}. Original ckpt: {}".format(
                    mask_channel_prob, model.mask_channel_prob
                )
            )
            model.mask_channel_prob = mask_channel_prob
        if mask_prob != model.mask_prob:
            logging.warning(
                "Overwriting mask prob to {}. Original ckpt to {}".format(
                    mask_prob, model.mask_prob
                )
            )
            model.mask_prob = mask_prob

        # CNN feature extractor is frozen during finetuning!
        self.encoders.feature_grad_mult = 0.0

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

        if subsample_output:
            self.subsample = get_subsample_module(
                subsample_mode,
                input_size=self.hubert_output_size,
                output_size=output_size,
            )
            self.subsample_mode = subsample_mode
            logging.info("Subsample output!")
            logging.info(self.subsample)
        else:
            self.subsample = None
            self.subsample_mode = None
            logging.info("Do not subsample output")

        self.training = training
        delattr(self.encoders, "final_proj")
        delattr(self.encoders, "label_embs_concat")

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--hubert-model-dir",
            type=str,
            help="Path to the pretrained Hubert model",
        )

        parser.add_argument(
            "--hubert-freeze-finetune-updates",
            type=int,
            default=0,
            help="The number of updates during which the transformer blocks \
                in hubert are frozen.",
        )

        parser.add_argument(
            "--hubert-mask-prob",
            type=float,
            default=0.65,
            help="Mask probability",
        )

        parser.add_argument(
            "--hubert-mask-channel-prob",
            type=float,
            default=0.5,
            help="Mask channel probability",
        )

        parser.add_argument(
            "--hubert-mask-channel-length",
            type=int,
            default=64,
            help="Mask channel length",
        )

        parser.add_argument(
            "--hubert-subsample-output",
            type=str2bool,
            default=False,
            help="Whether subsample the hubert output to reduce frame rate",
        )

        parser.add_argument(
            "--hubert-subsample-mode",
            type=str,
            default="concat_tanh",
            choices=["concat", "concat_relu", "concat_tanh", "avgpooling"],
        )

    def normalise_input(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize a batch of audio samples to zero mean unit variance

        Args:
            x (torch.Tensor):
                a batch of zero-padded audio samples shape: (B,T)
            x_lens (torch.Tensor):
                length of each audio sample in x, shape: (B)

        Returns:
            torch.Tensor: normalized audio samples, zero-padded
        """
        row_mean = x.sum(dim=-1, keepdim=True) / x_lens
        row_std = (
            torch.tensor(
                [sample[: x_lens[i]].std() for i, sample in enumerate(x)]
            )
            .view(x.size(0), 1)
            .to(x.device)
        )
        for i in range(x.size(0)):
            x[i, : x_lens[i]] = (x[i, : x_lens[i]] - row_mean[i]) / row_std[i]
        return x

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function of Fairseq Hubert Encoder

        Args:
            x (torch.Tensor):
                Input padded tensor of shape (B,T)
            x_lens (torch.Tensor):
                The number of frames of input (x) before padding
            warmup (float, optional):
                A floating point value that gradually increases from 0
                throughout training; when it is >= 1.0 we are "fully
                warmed up".  It is used to turn modules on sequentially.
                Actually not used in this function, just for consistency.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            A tuple of two tensors: Embedding and Embedding's length
        """

        mask = make_pad_mask(x_lens).to(x.device)
        x = self.normalise_input(x, x_lens)

        # check if still in freezing mode
        ft = (
            self.freeze_finetune_updates <= self.num_updates
        ) and self.training

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1

        if ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.warning(
                f"Start finetuning encoder after {self.num_updates} updates!"
            )

        # only apply modified specaug during training
        apply_mask = self.training
        with torch.set_grad_enabled(bool(ft)):
            enc_outputs = self.encoders(
                x,
                padding_mask=mask,
                features_only=True,
                mask=apply_mask,
            )
        xs_pad = enc_outputs["x"]
        masks = enc_outputs["padding_mask"]
        olens = torch.logical_not(masks).sum(dim=1)

        if self.subsample is not None:
            if "concat" in self.subsample_mode:
                xs_pad_1 = xs_pad[:, 0:-1:2, :]
                xs_pad_2 = xs_pad[:, 1::2, :]
                xs_pad = torch.cat((xs_pad_1, xs_pad_2), dim=2)
                xs_pad = self.subsample(xs_pad)
                olens = torch.floor(olens / 2)
            else:
                xs_pad = self.subsample(xs_pad.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                olens = torch.floor(olens / 2)

        return xs_pad, olens


def get_subsample_module(
    subsample_mode: str, input_size: int, output_size: int
) -> torch.nn.Module:
    """A subsample module to be added at the end of Hubert model

    Args:
        subsample_mode (str):
            Which subsample module to be selected
        input_size (int):
            Input size to the subsample module
        output_size (int):
            Output size of the subsample module

    Raises:
        NotImplementedError: _description_

    Returns:
        torch.nn.Module: _description_
    """
    if subsample_mode == "concat":
        subsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * input_size, output_size),
        )
    elif subsample_mode == "concat_relu":
        subsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * input_size, output_size),
            torch.nn.ReLU(),
        )
    elif subsample_mode == "concat_tanh":
        subsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * input_size, output_size),
            torch.nn.Tanh(),
        )
    else:
        raise NotImplementedError(
            "Only support: concat, concat_relu, concat_tanh, avgpooling"
        )

    return subsample
