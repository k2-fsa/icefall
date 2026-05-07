# Copyright      2025  Yifan Yang
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


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from icefall.utils import make_pad_mask


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class CLAP(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_downsample: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        text_encoder_dim: int = 768,
        joint_dim: int = 512,
    ):
        """A CLAP model.

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
        """
        super().__init__()

        # audio branch
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_downsample = encoder_downsample
        self.audio_projection = nn.Sequential(
            nn.Linear(encoder_dim, joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, joint_dim),
        )
        self.audio_transform = MLPLayers(
            units=[joint_dim, joint_dim, joint_dim], dropout=0.1
        )

        # text branch
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_projection = nn.Sequential(
            nn.Linear(text_encoder_dim, joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, joint_dim),
        )
        self.text_transform = MLPLayers(
            units=[joint_dim, joint_dim, joint_dim], dropout=0.1
        )

        self.logit_scale = nn.Parameter(torch.full((), math.log(1 / 0.07)))

    def forward_audio_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, freeze_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute audio encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        with torch.set_grad_enabled(not freeze_encoder):
            x, x_lens = self.encoder_embed(x, x_lens)
            src_key_padding_mask = make_pad_mask(x_lens)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            encoder_out, encoder_out_lens = self.encoder(
                x, x_lens, src_key_padding_mask
            )
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        if self.encoder_downsample is not None:
            encoder_out = encoder_out.permute(1, 0, 2)
            encoder_out = self.encoder_downsample(encoder_out)
            encoder_out = encoder_out.permute(1, 0, 2)
            encoder_out_lens = (encoder_out_lens + 1) // 2

        padding_mask = make_pad_mask(encoder_out_lens)
        encoder_out = encoder_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        embedding = encoder_out.sum(dim=1) / encoder_out_lens.unsqueeze(-1)  # (N, C)

        return embedding

    def forward_text_encoder(self, y: dict, freeze_encoder: bool = False):
        with torch.set_grad_enabled(not freeze_encoder):
            encoder_out = self.text_encoder(
                input_ids=y["input_ids"],
                attention_mask=y["attention_mask"],
            )["pooler_output"]

        return encoder_out

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        text: Optional[dict] = None,
        freeze_audio_encoder: bool = False,
        freeze_text_encoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          audio:
            A 3-D tensor of shape (N, T, C).
          audio_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          text:
            A dict containing the text input ids and attention mask.
        Returns:
          Return the CLAP loss
        """
        if audio is not None:
            assert audio.ndim == 3, audio.shape
            assert audio_lens.ndim == 1, audio_lens.shape

            audio_encoder_out = self.forward_audio_encoder(
                audio, audio_lens, freeze_encoder=freeze_audio_encoder
            )
            audio_encoder_out = self.audio_projection(audio_encoder_out)
            audio_encoder_out = self.audio_transform(audio_encoder_out)
            audio_encoder_out = F.normalize(audio_encoder_out, dim=-1)

        if text is not None:
            assert text["input_ids"].ndim == 2, text["input_ids"].shape

            text_encoder_out = self.forward_text_encoder(
                text, freeze_encoder=freeze_text_encoder
            )
            text_encoder_out = self.text_projection(text_encoder_out)
            text_encoder_out = self.text_transform(text_encoder_out)
            text_encoder_out = F.normalize(text_encoder_out, dim=-1)

        return (
            audio_encoder_out if audio is not None else None,
            text_encoder_out if text is not None else None,
            self.logit_scale.exp(),
        )
