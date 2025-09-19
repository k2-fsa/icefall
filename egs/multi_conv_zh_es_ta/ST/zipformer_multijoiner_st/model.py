# Copyright      2025  Johns Hopkins University (author: Amir Hussein)
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

from typing import Optional, Tuple, Union

import k2
import torch
from torch import Tensor
from lhotse.dataset import SpecAugment
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import  add_sos, make_pad_mask, time_warp
from scaling import ScaledLinear


class StModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        st_joiner: Optional[nn.Module] = None,
        st_decoder: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        st_vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_st_ctc: bool = False,
        use_hat: bool = False,
    ):
        """A multitask Transducer ASR-ST model with seperate joiners and predictors but shared acoustic encoder.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

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
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.use_hat = use_hat
        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner
            
            self.st_joiner = st_joiner
            self.st_decoder = st_decoder

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )

            self.simple_st_am_proj = ScaledLinear(
                encoder_dim, st_vocab_size, initial_scale=0.25
            )
            self.simple_st_lm_proj = ScaledLinear(
                decoder_dim, st_vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        self.use_st_ctc = use_st_ctc
        if self.use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
        if self.use_st_ctc:
            # Modules for CTC head
            self.st_ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, st_vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
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
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        return encoder_out, encoder_out_lens
    def forward_st_ctc(
        self,
        st_encoder_out: torch.Tensor,
        st_encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.st_ctc_output(st_encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=st_encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss
      
    def forward_cr_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTC loss with consistency regularization loss.
        Args:
          encoder_out:
            Encoder output, of shape (2 * N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (2 * N,).
          targets:
            Target Tensor of shape (2 * sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC loss
        ctc_output = self.ctc_output(encoder_out)  # (2 * N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="none",
        )
        ctc_loss_is_finite = torch.isfinite(ctc_loss)
        ctc_loss = ctc_loss[ctc_loss_is_finite]
        ctc_loss = ctc_loss.sum()

        # Compute consistency regularization loss
        exchanged_targets = ctc_output.detach().chunk(2, dim=0)
        exchanged_targets = torch.cat(
            [exchanged_targets[1], exchanged_targets[0]], dim=0
        )  # exchange: [x1, x2] -> [x2, x1]
        cr_loss = nn.functional.kl_div(
            input=ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )  # (2 * N, T, C)
        length_mask = make_pad_mask(encoder_out_lens).unsqueeze(-1)
        cr_loss = cr_loss.masked_fill(length_mask, 0.0).sum()

        return ctc_loss, cr_loss

    def forward_st_cr_ctc(
        self,
        st_encoder_out: torch.Tensor,
        st_encoder_out_lens: torch.Tensor,
        st_targets: torch.Tensor,
        st_target_lengths: torch.Tensor,
        # encoder_out: torch.Tensor,
        # encoder_out_lens: torch.Tensor,
        # targets: torch.Tensor,
        # target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTC loss with consistency regularization loss.
        Args:
          encoder_out:
            Encoder output, of shape (2 * N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (2 * N,).
          targets:
            Target Tensor of shape (2 * sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC loss
        st_ctc_output = self.st_ctc_output(st_encoder_out)  # (2 * N, T, C)
        st_ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=st_ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=st_targets.cpu(),
            input_lengths=st_encoder_out_lens.cpu(),
            target_lengths=st_target_lengths.cpu(),
            reduction="none",
        )
        st_ctc_loss_is_finite = torch.isfinite(st_ctc_loss)
        st_ctc_loss = st_ctc_loss[st_ctc_loss_is_finite]
        st_ctc_loss = st_ctc_loss.sum()
        # ctc_output = self.ctc_output(encoder_out)  # (2 * N, T, C)
        # ctc_loss = torch.nn.functional.ctc_loss(
        #     log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
        #     targets=targets.cpu(),
        #     input_lengths=encoder_out_lens.cpu(),
        #     target_lengths=target_lengths.cpu(),
        #     reduction="sum",
        # )
        
        # if not torch.isfinite(st_ctc_loss):
        #             breakpoint()

        # Compute consistency regularization loss
        exchanged_targets = st_ctc_output.detach().chunk(2, dim=0)
        exchanged_targets = torch.cat(
            [exchanged_targets[1], exchanged_targets[0]], dim=0
        )  # exchange: [x1, x2] -> [x2, x1]
        cr_loss = nn.functional.kl_div(
            input=st_ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )  # (2 * N, T, C)
        length_mask = make_pad_mask(st_encoder_out_lens).unsqueeze(-1)
        cr_loss = cr_loss.masked_fill(length_mask, 0.0).sum()

        return st_ctc_loss, cr_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        st_y: k2.RaggedTensor,
        st_y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        

        blank_id = self.decoder.blank_id
        st_blank_id = self.st_decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)
        st_sos_y = add_sos(st_y, sos_id=st_blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        st_sos_y_padded = st_sos_y.pad(mode="constant", padding_value=st_blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)
        st_decoder_out = self.st_decoder(st_sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        st_y_padded = st_y.pad(mode="constant", padding_value=0)

        st_y_padded = st_y_padded.to(torch.int64)
        st_boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        st_boundary[:, 2] = st_y_lens
        st_boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)
        st_lm = self.simple_st_lm_proj(st_decoder_out)
        st_am = self.simple_st_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
          simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
              lm=lm.float(),
              am=am.float(),
              symbols=y_padded,
              termination_symbol=blank_id,
              lm_only_scale=lm_scale,
              am_only_scale=am_scale,
              boundary=boundary,
              reduction="sum",
              return_grad=True,
          )
          st_simple_loss, (st_px_grad, st_py_grad) = k2.rnnt_loss_smoothed(
              lm=st_lm.float(),
              am=st_am.float(),
              symbols=st_y_padded,
              termination_symbol=st_blank_id,
              lm_only_scale=lm_scale,
              am_only_scale=am_scale,
              boundary=st_boundary,
              reduction="sum",
              return_grad=True,
          )
          

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
       
        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )
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
              reduction="sum",
              use_hat_loss=self.use_hat,
          )
         # logits : [B, T, prune_range, vocab_size]
        
        st_ranges = k2.get_rnnt_prune_ranges(
          px_grad=st_px_grad,
          py_grad=st_py_grad,
          boundary=st_boundary,
          s_range=prune_range,
        )
        st_am_pruned, st_lm_pruned = k2.do_rnnt_pruning(
            am=self.st_joiner.encoder_proj(encoder_out),
            lm=self.st_joiner.decoder_proj(st_decoder_out),
            ranges=st_ranges,
        )

        st_logits = self.st_joiner(st_am_pruned, st_lm_pruned, project_input=False)
        # Compute HAT loss for st
        with torch.cuda.amp.autocast(enabled=False):
          pruned_st_loss = k2.rnnt_loss_pruned(
              logits=st_logits.float(),
              symbols=st_y.pad(mode="constant", padding_value=blank_id).to(torch.int64),
              ranges=st_ranges,
              termination_symbol=st_blank_id,
              boundary=st_boundary,
              reduction="sum",
              use_hat_loss=self.use_hat,
          )
        
        return simple_loss, st_simple_loss, pruned_loss, pruned_st_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        st_y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_st_cr_ctc: bool = False,
        use_asr_cr_ctc: bool = False,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          use_cr_ctc:
            Whether use consistency-regularized CTC.
          use_spec_aug:
            Whether apply spec-augment manually, used only if use_cr_ctc is True.
          spec_augment:
            The SpecAugment instance that returns time masks,
            used only if use_cr_ctc is True.
          supervision_segments:
            An int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features``.
            Used only if use_cr_ctc is True.
          time_warp_factor:
            Parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
            Used only if use_cr_ctc is True.
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes
        assert st_y.num_axes == 2, st_y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        if use_st_cr_ctc or use_asr_cr_ctc:
          assert self.use_ctc or self.use_st_ctc
          if use_spec_aug:
              assert spec_augment is not None and spec_augment.time_warp_factor < 1
              # Apply time warping before input duplicating
              assert supervision_segments is not None
              x = time_warp(
                  x,
                  time_warp_factor=time_warp_factor,
                  supervision_segments=supervision_segments,
              )
              # Independently apply frequency masking and time masking to the two copies
              x = spec_augment(x.repeat(2, 1, 1))
          else:
              x = x.repeat(2, 1, 1)
          x_lens = x_lens.repeat(2)
          y = k2.ragged.cat([y, y], axis=0)
          if self.st_joiner != None and self.use_st_ctc:
            st_y = k2.ragged.cat([st_y, st_y], axis=0)

        # Compute encoder outputs

        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]
        st_row_splits = st_y.shape.row_splits(1)
        st_y_lens = st_row_splits[1:] - st_row_splits[:-1]

        if self.use_transducer:
            
            # Compute transducer loss
            if self.st_joiner != None:
              simple_loss, st_simple_loss, pruned_loss, st_pruned_loss = self.forward_transducer(
                  encoder_out=encoder_out,
                  encoder_out_lens=encoder_out_lens,
                  y=y.to(x.device),
                  y_lens=y_lens,
                  st_y=st_y.to(x.device),
                  st_y_lens=st_y_lens,
                  prune_range=prune_range,
                  am_scale=am_scale,
                  lm_scale=lm_scale,
              )
              if use_asr_cr_ctc:
                simple_loss = simple_loss * 0.5
                pruned_loss = pruned_loss * 0.5
              if use_st_cr_ctc:
                st_simple_loss = st_simple_loss * 0.5
                st_pruned_loss = st_pruned_loss * 0.5
            else:
                simple_loss, pruned_loss = self.forward_transducer(
                  encoder_out=encoder_out,
                  encoder_out_lens=encoder_out_lens,
                  y=y.to(x.device),
                  y_lens=y_lens,
                  prune_range=prune_range,
                  am_scale=am_scale,
                  lm_scale=lm_scale,
              )
                if use_asr_cr_ctc:
                  simple_loss = simple_loss * 0.5
                  pruned_loss = pruned_loss * 0.5
                st_simple_loss, st_pruned_loss = torch.empty(0), torch.empty(0)
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            if not use_asr_cr_ctc:
                ctc_loss = self.forward_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
                cr_loss = torch.empty(0)
            else:
                ctc_loss, cr_loss = self.forward_cr_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
                ctc_loss = ctc_loss * 0.5
                cr_loss = cr_loss * 0.5
        else:
            cr_loss = torch.empty(0)
            ctc_loss = torch.empty(0)
        if self.use_st_ctc:
            st_targets = st_y.values
            if not use_st_cr_ctc:
              st_ctc_loss = self.forward_st_ctc(
                    st_encoder_out=encoder_out,
                    st_encoder_out_lens=encoder_out_lens,
                    targets=st_targets,
                    target_lengths=st_y_lens,
                )
              st_cr_loss = torch.empty(0)
            else:
              st_ctc_loss, st_cr_loss = self.forward_st_cr_ctc(
                  st_encoder_out=encoder_out,
                  st_encoder_out_lens=encoder_out_lens,
                  st_targets=st_targets,
                  st_target_lengths=st_y_lens,
                  # encoder_out=encoder_out,
                  # encoder_out_lens=encoder_out_lens,
                  # targets=targets,
                  # target_lengths=y_lens,
              )
              st_ctc_loss = st_ctc_loss * 0.5
              st_cr_loss = st_cr_loss * 0.5
        else:
            st_ctc_loss = torch.empty(0)
            st_cr_loss = torch.empty(0)

        return simple_loss, st_simple_loss, pruned_loss, st_pruned_loss, ctc_loss, st_ctc_loss, cr_loss, st_cr_loss
