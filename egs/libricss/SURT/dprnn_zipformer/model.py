# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
# Copyright    2023  Johns Hopkins University (author: Desh Raj)
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

from typing import List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import add_sos


class SURT(nn.Module):
    """It implements Streaming Unmixing and Recognition Transducer (SURT).
    https://arxiv.org/abs/2011.13148
    """

    def __init__(
        self,
        mask_encoder: nn.Module,
        encoder: EncoderInterface,
        joint_encoder_layer: Optional[nn.Module],
        decoder: nn.Module,
        joiner: nn.Module,
        num_channels: int,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          mask_encoder:
            It is the masking network. It generates a mask for each channel of the
            encoder. These masks are applied to the input features, and then passed
            to the transcription network.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
          num_channels:
            It is the number of channels that the input features will be split into.
            In general, it should be equal to the maximum number of simultaneously
            active speakers. For most real scenarios, using 2 channels is sufficient.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.mask_encoder = mask_encoder
        self.encoder = encoder
        self.joint_encoder_layer = joint_encoder_layer
        self.decoder = decoder
        self.joiner = joiner
        self.num_channels = num_channels

        self.simple_am_proj = nn.Linear(
            encoder_dim,
            vocab_size,
        )
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward_helper(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        reduction: str = "sum",
        beam_size: int = 10,
        use_double_scores: bool = False,
        subsampling_factor: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute transducer loss for one branch of the SURT model.
        """
        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        if self.joint_encoder_layer is not None:
            encoder_out = self.joint_encoder_layer(encoder_out)

        # compute ctc log-probs
        ctc_output = self.ctc_output(encoder_out)

        # For the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction=reduction,
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction=reduction,
            )

        # Compute ctc loss
        supervision_segments = torch.stack(
            (
                torch.arange(len(x_lens), device="cpu"),
                torch.zeros_like(x_lens, device="cpu"),
                torch.clone(x_lens).detach().cpu(),
            ),
            dim=1,
        ).to(torch.int32)
        # We need to sort supervision_segments in decreasing order of num_frames
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        # Works with a BPE model
        decoding_graph = k2.ctc_graph(y, modified=False, device=x.device)
        dense_fsa_vec = k2.DenseFsaVec(
            ctc_output,
            supervision_segments,
            allow_truncate=subsampling_factor - 1,
        )
        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=beam_size,
            reduction="none",
            use_double_scores=use_double_scores,
        )

        return (simple_loss, pruned_loss, ctc_loss)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        reduction: str = "sum",
        beam_size: int = 10,
        use_double_scores: bool = False,
        subsampling_factor: int = 1,
        return_masks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor of shape (N*num_channels, S). It contains the labels
            of the N utterances. The labels are in the range [0, vocab_size). All
            the channels are concatenated together one after another.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
          beam_size:
            The beam size used in CTC decoding.
          use_double_scores:
            If True, use double precision for CTC decoding.
          subsampling_factor:
            The subsampling factor of the model. It is used to compute the
            supervision segments for CTC loss.
          return_masks:
            If True, return the masks as well as masked features.
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0), (x.size(), x_lens.size())

        # Apply the mask encoder
        B, T, F = x.shape
        processed = self.mask_encoder(x)  # B,T,F*num_channels
        masks = processed.view(B, T, F, self.num_channels).unbind(dim=-1)
        x_masked = [x * m for m in masks]

        # Recognition
        # Stack the inputs along the batch axis
        h = torch.cat(x_masked, dim=0)
        h_lens = torch.cat([x_lens for _ in range(self.num_channels)], dim=0)

        simple_loss, pruned_loss, ctc_loss = self.forward_helper(
            h,
            h_lens,
            y,
            prune_range,
            am_scale,
            lm_scale,
            reduction=reduction,
            beam_size=beam_size,
            use_double_scores=use_double_scores,
            subsampling_factor=subsampling_factor,
        )

        # Chunks the outputs into 2 parts along batch axis and then stack them along a new axis.
        simple_loss = torch.stack(
            torch.chunk(simple_loss, self.num_channels, dim=0), dim=0
        )
        pruned_loss = torch.stack(
            torch.chunk(pruned_loss, self.num_channels, dim=0), dim=0
        )
        ctc_loss = torch.stack(torch.chunk(ctc_loss, self.num_channels, dim=0), dim=0)

        if return_masks:
            return (simple_loss, pruned_loss, ctc_loss, x_masked, masks)
        else:
            return (simple_loss, pruned_loss, ctc_loss, x_masked)
