import logging
from typing import Tuple

import fairseq
import torch
from encoder_interface import EncoderInterface

from icefall.utils import make_pad_mask


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

        self.encoders.feature_grad_mult = (
            0.0  # CNN feature extractor is frozen during finetuning!
        )
        # self.pretrained_params = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

        if subsample_output:
            self.subsample = get_subsample_module(
                subsample_mode,
                input_size=self.hubert_output_size,
                output_size=output_size,
            )
            self.subsample_mode = subsample_mode
        else:
            self.subsample = None
            self.subsample_mode = None

        self.training = training
        delattr(self.encoders, "final_proj")
        delattr(self.encoders, "label_embs_concat")
        # self.conv_subsampling_factor = 1 # ESPnet required

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Fairseq Hubert Encoder

        Args:
            xs_pad (torch.Tensor):
                Input padded tensor of shape (B,T,C)
            ilens (torch.Tensor):
                The number of frames of input (xs_pad) before padding
            warmup (float, optional):
                A floating point value that gradually increases from 0
                throughout training; when it is >= 1.0 we are "fully
                warmed up".  It is used to turn modules on sequentially.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors:
            Embedding and Embedding's length
        """

        mask = make_pad_mask(x_lens).to(x.device)
        row_mean = (x.sum(dim=-1) / x_lens).view(x.size(0), 1)
        row_std = (
            torch.tensor([sample[sample != 0.0].std() for sample in x])
            .view(x.size(0), 1)
            .to(x.device)
        )
        for i in range(x.size(0)):
            x[i, : x_lens[i]] = (x[i, : x_lens[i]] - row_mean[i]) / row_std[i]

        # check if still in freezing mode
        ft = (
            self.freeze_finetune_updates <= self.num_updates
        ) and self.training

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.warning(
                "Start fine-tuning Hubert parameters after {} updates!".format(
                    self.num_updates
                )
            )

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

        if self.subsample:
            if "concat" in self.subsample_mode:
                xs_pad_1 = xs_pad[:, 0:-1:2, :]
                xs_pad_2 = xs_pad[:, 1::2, :]
                xs_pad = torch.cat((xs_pad_1, xs_pad_2), dim=2)
                xs_pad = self.subsample(xs_pad)
                olens = olens // 2
            else:
                xs_pad = self.subsample(xs_pad.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                olens = olens // 2

        return xs_pad, olens


def get_subsample_module(subsample_mode, input_size, output_size):
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
    elif subsample_mode == "avgpooling":
        subsample = torch.nn.Sequential(
            torch.nn.AvgPool1d(kernel_size=2, stride=2)
        )
    else:
        raise NotImplementedError(
            "Only support: concat, concat_relu, concat_tanh, avgpooling"
        )

    return subsample
