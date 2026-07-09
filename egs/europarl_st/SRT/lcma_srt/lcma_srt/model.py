# -*- coding: utf-8 -*-

# Copyright 2025 Nanjie Li (linanjie0820@gmail.com)
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

from typing import Optional, Tuple, List

import k2
import torch
import torch.nn as nn
from torch import Tensor, nn
import warnings
from encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment
from scaling import ScaledLinear

from icefall.utils import add_sos, make_pad_mask, time_warp

from moe_adapter import MoEAdapterDense
class LCMASRTModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        enc_asr: EncoderInterface,
        enc_st: EncoderInterface,
        decoder_asr: Optional[nn.Module] = None,
        joiner_asr: Optional[nn.Module] = None,
        attention_decoder_asr: Optional[nn.Module] = None,
        decoder_st: Optional[nn.Module] = None,
        joiner_st: Optional[nn.Module] = None,
        attention_decoder_st: Optional[nn.Module] = None,
        encoder_dim_asr: int = 384,
        encoder_dim_st: int = 384,
        decoder_dim_asr: int = 512,
        decoder_dim_st: int = 512,
        vocab_size_asr: int = 500,
        vocab_size_st: int = 6000,
        output_downsampling_factor_asr: int = 2,
        output_downsampling_factor_st: int = 2,
        num_srt_langs_asr: int = 0,
        num_tgt_langs_ast: int = 0,
        num_experts_asr: int = 4,
        num_experts_ast: int = 8,
        entropy_reg_asr: float = 0.0,
        entropy_reg_ast: float = 0.0,
        temperature_asr: float = 1.0,
        temperature_ast: float = 1.0,
        asr_moe: bool = True,
        asr_src: bool = True,
        ast_moe: bool = True,
        ast_tgt: bool = True,
        # task flags
        use_transducer: bool = True,
        use_ctc_asr: bool = True,
        use_ctc_st: bool = True,
        # CR-CTC / Attn flags (CR-CTC is supported; attn optional)
        use_attention_decoder: bool = False,
        freeze_asr: bool = False,
        freeze_frontend: bool = False,
    ):
        super().__init__()
        assert (
            use_transducer or use_ctc_asr
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc_asr}"
        assert isinstance(enc_asr, EncoderInterface)
        assert isinstance(enc_st, EncoderInterface)
        self.is_srt = True
        self.encoder_embed = encoder_embed
        self.enc_asr = enc_asr
        self.enc_st = enc_st

        self.use_transducer = use_transducer
        self.use_ctc_asr = use_ctc_asr
        self.use_ctc_st = use_ctc_st
        self.use_attention_decoder = use_attention_decoder

        self.output_downsampling_factor_asr=output_downsampling_factor_asr
        self.output_downsampling_factor_st=output_downsampling_factor_st
        # ------ ASR head ------
        if self.use_transducer:
            assert decoder_asr is not None and joiner_asr is not None
            assert hasattr(decoder_asr, "blank_id")
            self.decoder_asr = decoder_asr
            self.joiner_asr = joiner_asr
            self.simple_am_proj_asr = ScaledLinear(encoder_dim_asr, vocab_size_asr, initial_scale=0.25)
            self.simple_lm_proj_asr = ScaledLinear(decoder_dim_asr, vocab_size_asr, initial_scale=0.25)
        else:
            assert decoder_asr is None
            assert joiner_asr is None

        if self.use_ctc_asr:
            self.ctc_asr = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim_asr, vocab_size_asr),
                nn.LogSoftmax(dim=-1),
            )

        # ------ ST head ------
        if self.use_transducer:
            assert decoder_st is not None and joiner_st is not None
            assert hasattr(decoder_st, "blank_id")
            self.decoder_st = decoder_st
            self.joiner_st = joiner_st
            self.simple_am_proj_st = ScaledLinear(encoder_dim_st, vocab_size_st, initial_scale=0.25)
            self.simple_lm_proj_st = ScaledLinear(decoder_dim_st, vocab_size_st, initial_scale=0.25)
        else:
            assert decoder_st is None
            assert joiner_st is None
        if self.use_ctc_st:
            self.ctc_st = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim_st, vocab_size_st),
                nn.LogSoftmax(dim=-1),
            )

        # optional attention decoder heads could be added similarly if needed
        if use_attention_decoder:
            self.attention_decoder_asr = attention_decoder_asr
            self.attention_decoder_st = attention_decoder_st
        else:
            assert attention_decoder_asr is None
            assert attention_decoder_st is None

        self.freeze_asr = freeze_asr
        self.freeze_frontend = freeze_frontend
        if self.freeze_asr:
            self._apply_freeze_asr()

        self.asr_moe = asr_moe
        self.asr_src = asr_src
        self.ast_moe = ast_moe
        self.ast_tgt = ast_tgt

        self.asr_moe_layer: Optional[MoEAdapterDense] = None
        if self.asr_moe:
            num_langs_asr = num_srt_langs_asr if self.asr_src else 0
            self.asr_moe_layer = MoEAdapterDense(
                d_model=enc_asr.output_dim,
                num_experts=num_experts_asr,
                hidden_mult=1.3,
                num_tasks=0,   
                num_langs=num_langs_asr,
                dropout=0.1,
                entropy_reg=entropy_reg_asr,  
                temperature=temperature_asr,
            )
        self.lang_embed_asr = (
            nn.Embedding(num_srt_langs_asr, enc_asr.output_dim) if (not self.asr_moe and self.asr_src) else None
        )


        self.ast_moe_layer: Optional[MoEAdapterDense] = None
        if self.ast_moe:
            num_langs_ast = num_tgt_langs_ast if self.ast_tgt else 0
            self.ast_moe_layer = MoEAdapterDense(
                d_model=enc_st.output_dim,
                num_experts=num_experts_ast,
                hidden_mult=1.3,
                num_tasks=0,
                num_langs=num_langs_ast,
                num_src_langs=0,
                num_tgt_langs=0,
                dropout=0.1,
                entropy_reg=entropy_reg_ast,
                temperature=temperature_ast,
            )
        self.lang_embed_ast = (
            nn.Embedding(num_tgt_langs_ast, enc_st.output_dim) if (not self.ast_moe and self.ast_tgt) else None
        )

    def _apply_freeze_asr(self):
        to_freeze = []
        if self.freeze_frontend and hasattr(self, "encoder_embed"):
            to_freeze += [self.encoder_embed]
        to_freeze += [self.enc_asr]
        for name in ["decoder_asr", "joiner_asr", "ctc_asr", "attention_decoder_asr","simple_am_proj_asr", "simple_lm_proj_asr"]:
            if hasattr(self, name) and getattr(self, name) is not None:
                to_freeze += [getattr(self, name)]

        for m in to_freeze:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def output_downsampling(self,x_lengths):
        # class Downsample has this rounding behavior..
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            lengths = (x_lengths + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lengths + 1) // 2

        return lengths


    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        srt_lang_ids: Optional[torch.Tensor] = None,
        tgt_lang_ids: Optional[torch.Tensor] = None,
        enable_st: bool = True,
        return_moe_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
        """

        if self.freeze_frontend:
            with torch.no_grad():
                x, x_lens = self.encoder_embed(x, x_lens)
        else:
            x, x_lens = self.encoder_embed(x, x_lens)
            
        pad0 = make_pad_mask(x_lens)
        x_tnc = x.permute(1, 0, 2)                    # (T,N,C)

        if self.freeze_asr:
            with torch.no_grad():
                asr_x, asr_len = self.enc_asr(x_tnc, x_lens, pad0)
        else:
            asr_x, asr_len = self.enc_asr(x_tnc, x_lens, pad0) # (T1,N,C1), (N,)

        if self.freeze_asr:
            asr_x = asr_x.detach()        
        
        w_asr: Optional[torch.Tensor] = None
        if self.asr_moe and self.asr_moe_layer is not None:
            if self.asr_moe_layer.lang_embed is not None:
                if srt_lang_ids is None:
                    raise ValueError("srt_lang_ids is required when asr_moe=True and asr_src=True")
                lang_ids = srt_lang_ids.to(dtype=torch.long, device=asr_x.device)
            else:
                lang_ids = None
            asr_x_for_asr, w_asr = self.asr_moe_layer(asr_x, lang_ids=lang_ids)
        elif self.asr_src and self.lang_embed_asr is not None:
            if srt_lang_ids is None:
                raise ValueError("srt_lang_ids is required when asr_src=True and asr_moe=False")
            srt_lang_ids = srt_lang_ids.to(dtype=torch.long, device=asr_x.device)
            lang_bias = self.lang_embed_asr(srt_lang_ids).unsqueeze(0).to(dtype=asr_x.dtype)
            asr_x_for_asr, w_asr = asr_x + lang_bias, None
        else:
            asr_x_for_asr, w_asr = asr_x, None

        asr_tnc = self.enc_asr.downsample_output(asr_x_for_asr)
        asr_output_len = self.output_downsampling(asr_len)
        asr_ntc = asr_tnc.permute(1, 0, 2)

        moe_terms: List[torch.Tensor] = []
        if self.asr_moe and w_asr is not None and self.asr_moe_layer is not None:
            pad_tb_asr = make_pad_mask(asr_len).transpose(0, 1)
            moe_ent_asr = self.asr_moe_layer.router_entropy_loss(w=w_asr, pad_mask_tb=pad_tb_asr)
            moe_terms.append(moe_ent_asr)

        w_ast: Optional[torch.Tensor] = None
        if not enable_st:
            dummy_st = torch.empty(0, device=asr_ntc.device)
            dummy_lens = torch.zeros_like(asr_output_len)
            if moe_terms:
                moe_ent_loss = torch.stack(moe_terms).mean()
            else:
                moe_ent_loss = torch.zeros((), device=asr_ntc.device)
            if return_moe_weights:
                return (
                    asr_ntc,
                    asr_output_len,
                    dummy_st,
                    dummy_lens,
                    moe_ent_loss,
                    w_asr,
                    w_ast,
                )
            return asr_ntc, asr_output_len, dummy_st, dummy_lens, moe_ent_loss


        from scaling import convert_num_channels
        st_in = convert_num_channels(asr_x_for_asr, self.enc_st.encoder_dim[0])
        st_x, st_len = self.enc_st(st_in, asr_len, make_pad_mask(asr_len))
        
        
        if self.ast_moe and self.ast_moe_layer is not None:
            if self.ast_moe_layer.lang_embed is not None:
                if tgt_lang_ids is None:
                    raise ValueError("tgt_lang_ids is required when ast_moe=True and ast_tgt=True")
                lang_ids = tgt_lang_ids.to(dtype=torch.long, device=st_x.device)
            else:
                lang_ids = None
            ast_x_for_ast, w_ast = self.ast_moe_layer(
                st_x,
                lang_ids=lang_ids,
            )
        elif self.ast_tgt and self.lang_embed_ast is not None:
            if tgt_lang_ids is None:
                raise ValueError("tgt_lang_ids is required when ast_tgt=True and ast_moe=False")
            tgt_lang_ids = tgt_lang_ids.to(dtype=torch.long, device=st_x.device)
            lang_bias = self.lang_embed_ast(tgt_lang_ids).unsqueeze(0).to(dtype=st_x.dtype)
            ast_x_for_ast, w_ast = st_x + lang_bias, None
        else:
            ast_x_for_ast, w_ast = st_x, None

        if self.output_downsampling_factor_st == 2:
            st_tnc = self.enc_st.downsample_output(ast_x_for_ast)
            st_lens = self.output_downsampling(st_len)
        elif self.output_downsampling_factor_st == 1:
            st_tnc = ast_x_for_ast
            st_lens = st_len

        st_ntc = st_tnc.permute(1, 0, 2)              # (N,T2,C2)

        assert torch.all(asr_output_len > 0) and torch.all(st_lens > 0), (asr_output_len, st_lens)
        # return asr_ntc, asr_output_len, st_ntc, st_lens
        

        if self.ast_moe and w_ast is not None and self.ast_moe_layer is not None:
            pad_tb_st = make_pad_mask(st_len).transpose(0, 1)
            moe_ent_st = self.ast_moe_layer.router_entropy_loss(w=w_ast, pad_mask_tb=pad_tb_st)
            moe_terms.append(moe_ent_st)

        if moe_terms:
            moe_ent_loss = torch.stack(moe_terms).mean()
        else:
            moe_ent_loss = torch.zeros((), device=asr_ntc.device)

        if return_moe_weights:
            return (
                asr_ntc,
                asr_output_len,
                st_ntc,
                st_lens,
                moe_ent_loss,
                w_asr,
                w_ast,
            )

        return asr_ntc, asr_output_len, st_ntc, st_lens, moe_ent_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,  # (N, T, C)
        encoder_out_lens: torch.Tensor,  # (N,)
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        decoder: nn.Module,
        joiner: nn.Module,
        simple_lm_proj: nn.Module,
        simple_am_proj: nn.Module,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        blank_id = decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=blank_id)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = simple_lm_proj(decoder_out)
        am = simple_am_proj(encoder_out)

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
            am=joiner.encoder_proj(encoder_out),
            lm=joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_output: nn.Module,
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
        ctc_output = ctc_output(encoder_out)  # (N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward_cr_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_output: nn.Module,
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
        ctc_output = ctc_output(encoder_out)  # (2 * N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )

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

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y_asr: k2.RaggedTensor,
        y_st: Optional[k2.RaggedTensor] = None,
        srt_lang_ids: Optional[torch.Tensor] = None,
        tgt_lang_ids: Optional[torch.Tensor] = None,
        prune_range_asr: int = 5,
        prune_range_st: int = 10,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_cr_ctc: bool = True,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        enable_st: bool = True,
    ):

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y_asr.num_axes == 2, y_asr.num_axes
        if enable_st:
            assert y_st is not None
            assert y_st.num_axes == 2, y_st.num_axes
            assert x.size(0) == x_lens.size(0) == y_asr.dim0 == y_st.dim0, (
                x.shape, x_lens.shape, y_asr.dim0, y_st.dim0
            )
        else:
            assert x.size(0) == x_lens.size(0) == y_asr.dim0
        device = x.device

        if use_cr_ctc:
            assert self.use_ctc_asr
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
            y_asr = k2.ragged.cat([y_asr, y_asr], axis=0)

            if enable_st and (y_st is not None):
                y_st = k2.ragged.cat([y_st, y_st], axis=0)
            
            if srt_lang_ids is not None:
                srt_lang_ids = srt_lang_ids.repeat(2)
                srt_lang_ids = srt_lang_ids.to(x.device).view(-1)
            if tgt_lang_ids is not None and enable_st:
                tgt_lang_ids = tgt_lang_ids.repeat(2)
                tgt_lang_ids = tgt_lang_ids.to(x.device).view(-1)


        asr_ntc, asr_lens, st_ntc, st_lens, moe_ent_loss = self.forward_encoder(
            x,
            x_lens,
            srt_lang_ids=srt_lang_ids,
            tgt_lang_ids=tgt_lang_ids if enable_st else None,
            enable_st=enable_st,
        )

        
        # ASR prepare targets
        row_splits_asr = y_asr.shape.row_splits(1)
        y_lens_asr = row_splits_asr[1:] - row_splits_asr[:-1]

        # AST prepare targets
        if enable_st and (y_st is not None):
            row_splits_st = y_st.shape.row_splits(1)
            y_lens_st = row_splits_st[1:] - row_splits_st[:-1]
        else:
            y_lens_st = None

        # ---------- RNNT losses ----------
        if self.use_transducer:
            simple_loss_asr, pruned_loss_asr = self.forward_transducer(
                encoder_out=asr_ntc,
                encoder_out_lens=asr_lens,
                y=y_asr.to(device),
                y_lens=y_lens_asr,
                decoder=self.decoder_asr,
                joiner=self.joiner_asr,
                simple_lm_proj=self.simple_lm_proj_asr,
                simple_am_proj=self.simple_am_proj_asr,
                prune_range=prune_range_asr,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )

            if enable_st and (y_st is not None) and (y_lens_st is not None):
                simple_loss_st, pruned_loss_st = self.forward_transducer(
                    encoder_out=st_ntc,
                    encoder_out_lens=st_lens,
                    y=y_st.to(device),
                    y_lens=y_lens_st,
                    decoder=self.decoder_st,
                    joiner=self.joiner_st,
                    simple_lm_proj=self.simple_lm_proj_st,
                    simple_am_proj=self.simple_am_proj_st,
                    prune_range=prune_range_st,
                    am_scale=am_scale,
                    lm_scale=lm_scale,
                )
            else:
                simple_loss_st = pruned_loss_st = torch.empty(0)

            if use_cr_ctc:
                simple_loss_asr = simple_loss_asr * 0.5
                pruned_loss_asr = pruned_loss_asr * 0.5
                if simple_loss_st.numel() > 0:
                    simple_loss_st = simple_loss_st * 0.5
                    pruned_loss_st = pruned_loss_st * 0.5
        else:
            simple_loss_asr = pruned_loss_asr = torch.empty(0)
            simple_loss_st = pruned_loss_st = torch.empty(0)

        
        
        if self.use_ctc_asr:
            # Compute CTC loss
            targets_asr = y_asr.values

            if not use_cr_ctc:
                ctc_loss_asr = self.forward_ctc(
                    encoder_out=asr_ntc,
                    encoder_out_lens=asr_lens,
                    targets=targets_asr,
                    target_lengths=y_lens_asr,
                    ctc_output=self.ctc_asr,
                )
                cr_loss_asr = torch.empty(0)
            else:
                ctc_loss_asr, cr_loss_asr = self.forward_cr_ctc(
                    encoder_out=asr_ntc,
                    encoder_out_lens=asr_lens,
                    targets=targets_asr,
                    target_lengths=y_lens_asr,
                    ctc_output=self.ctc_asr,
                )
                ctc_loss_asr = ctc_loss_asr * 0.5
                cr_loss_asr = cr_loss_asr * 0.5
        else:
            ctc_loss_asr = cr_loss_asr = torch.empty(0)

        if self.use_ctc_st and enable_st and (y_st is not None) and (y_lens_st is not None):
            targets_st = y_st.values
            if not use_cr_ctc:
                ctc_loss_st = self.forward_ctc(
                    encoder_out=st_ntc,
                    encoder_out_lens=st_lens,
                    targets=targets_st,
                    target_lengths=y_lens_st,
                    ctc_output=self.ctc_st,
                )
                cr_loss_st = torch.empty(0)
            else:
                ctc_loss_st, cr_loss_st = self.forward_cr_ctc(
                    encoder_out=st_ntc,
                    encoder_out_lens=st_lens,
                    targets=targets_st,
                    target_lengths=y_lens_st,
                    ctc_output=self.ctc_st,
                )
                ctc_loss_st = ctc_loss_st * 0.5
                cr_loss_st = cr_loss_st * 0.5
        else:
            ctc_loss_st = cr_loss_st = torch.empty(0)

        if self.use_attention_decoder:
            attention_decoder_loss_asr = self.attention_decoder_asr.calc_att_loss(
                encoder_out=asr_ntc,
                encoder_out_lens=asr_lens,
                ys=y_asr.to(device),
                ys_lens=y_lens_asr.to(device),
            )
            if use_cr_ctc:
                attention_decoder_loss_asr = attention_decoder_loss_asr * 0.5

            if enable_st and (y_st is not None) and (y_lens_st is not None):
                attention_decoder_loss_st = self.attention_decoder_st.calc_att_loss(
                    encoder_out=st_ntc,
                    encoder_out_lens=st_lens,
                    ys=y_st.to(device),
                    ys_lens=y_lens_st.to(device),
                )
                if use_cr_ctc:
                    attention_decoder_loss_st = attention_decoder_loss_st * 0.5
            else:
                attention_decoder_loss_st = torch.empty(0)
        else:
            attention_decoder_loss_asr = torch.empty(0)
            attention_decoder_loss_st = torch.empty(0)


        return simple_loss_asr, simple_loss_st, pruned_loss_asr, pruned_loss_st, ctc_loss_asr, ctc_loss_st, attention_decoder_loss_asr, attention_decoder_loss_st, cr_loss_asr, cr_loss_st, moe_ent_loss
