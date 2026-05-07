# Copyright    2021-2023  Xiaomi Corp.        (authors: Xiaoyu Yang,
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
import random
from typing import List, Optional, Tuple

import k2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment

from icefall.utils import AttributeDict, make_pad_mask


class AudioPretrainingModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: nn.Module,
        fbank_dim: int = 80,
        encoder_dim: int = 384,
        encoder_input_dim: int = 192,
        decoder_dim: int = 384,
        decoder_input_dim: int = 192,
        noise_scale: float = 0.1,
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_selection: str = "static",
        mask_other: float = 0.0,
    ):
        """An audio pretraining model

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
          encoder_dim:
            Dimension of the encoder.
          noise_scale:
            The scale of the gaussia noise.
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.fbank_dim = fbank_dim
        
        self.decoder = decoder
        self.decoder_input_dim = decoder_input_dim
        self.decoder_dim = decoder_dim
        
        # decoder embed
        self.decoder_embed = nn.Linear(
            encoder_dim, decoder_input_dim, bias=True,
        )
        # decoder pred to 4 * fbank dim (we concatenate every 4 frames)
        self.decoder_pred = nn.Linear(
            decoder_dim, fbank_dim * 4, bias=True,
        )

        # mask embeddings
        self.mask_emb = nn.Parameter(torch.FloatTensor(fbank_dim).uniform_())
        self.decoder_mask_emb = nn.Parameter(torch.FloatTensor(encoder_dim).normal_())

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other

        self.noise_scale = noise_scale

    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
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
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          target:
            The reconstruction target
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        N, T, C = x.shape

        padding_mask = make_pad_mask(x_lens)

        # apply masking to the fbank features
        x, mask_indices = self.apply_mask_facebook(
            x.clone(),
            padding_mask=padding_mask
        ) # (N,T,C), (N,T)

        x, x_lens = self.encoder_embed(x, x_lens) # (N,T,C)
        src_key_padding_mask = make_pad_mask(x_lens)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask) # (T,N,C)

        # Normalize encoder features
        normalize_factor = (encoder_out ** 2).mean(dim=-1, keepdim=True).sqrt()
        encoder_out = encoder_out / normalize_factor

        if self.training:
            # add noise to the encoder_out
            noise = torch.randn_like(encoder_out, device=encoder_out.device) * self.noise_scale
            encoder_out += noise

        # replace the masked encoder_out with a mask_emb
        decoder_mask_indices = nn.functional.max_pool1d(mask_indices, 4)
        assert decoder_mask_indices.size(1) >= encoder_out.size(0)
        if decoder_mask_indices.size(1) > encoder_out.size(0):
            decoder_mask_indices = decoder_mask_indices[:, :encoder_out.size(0)]

        decoder_mask_indices = decoder_mask_indices.bool().T
        encoder_out[decoder_mask_indices] = self.decoder_mask_emb
            
        # perform the reconstruction
        decoder_src_key_padding_mask = make_pad_mask(encoder_out_lens)
        decoder_in = self.decoder_embed(encoder_out) # project to decoder_dim
        decoder_out, decoder_out_lens = self.decoder(decoder_in, encoder_out_lens, decoder_src_key_padding_mask)

        decoder_out = self.decoder_pred(decoder_out) 
        decoder_out = decoder_out.permute(1, 0, 2) # (T, N, C) -> (N, T, C)

        # compute the reconstruction loss
        assert target.size(1) >= 4 * decoder_out.size(1), (target.size(1), decoder_out.size(1))
        target = target[:, : 4 * decoder_out.size(1), :].reshape(N, -1, 4, self.fbank_dim) 
        target = target.reshape(N, -1, 4 * self.fbank_dim)
        l2_loss = nn.functional.mse_loss(
            decoder_out,
            target,
            reduction="none"
        ) # (N, T, C)

        # mask the loss on the padding positions
        l2_loss.masked_fill_(decoder_src_key_padding_mask.unsqueeze(-1), 0.0)
        
        # only compute reconstruction loss on masked frames
        mask_indices = nn.functional.max_pool1d(mask_indices.float(), 4)
        assert mask_indices.size(1) >= decoder_src_key_padding_mask.size(1)
        if mask_indices.size(1) > decoder_src_key_padding_mask.size(1):
            mask_indices = mask_indices[:, :decoder_src_key_padding_mask.size(1)]
        l2_loss *= mask_indices.unsqueeze(-1)
        
        # normalize the mse loss by the fbank dimension 
        l2_loss = l2_loss.sum() / self.fbank_dim 

        return l2_loss
    
    def apply_mask_facebook(
        self,
        x: torch.Tensor,
        padding_mask,
    ):
        # this function is modified from fairseq: https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/models/wav2vec/wav2vec2.py#L429
        # The masked indices have value 1
        B, T, C = x.shape

        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=2,
                no_overlap=False,  # False
                min_space=1,  # 1
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
            mask_indices = mask_indices.float()
            if random.random() > 0.97:
                logging.info(f"A proportion of {mask_indices.sum()/mask_indices.numel():.2f} frames are masked")
        else:
            mask_indices = None

        return x, mask_indices


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError("this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask

