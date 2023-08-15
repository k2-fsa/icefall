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

import warnings
from typing import List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos, encode_supervisions, make_pad_mask


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        lconv: Optional[nn.Module] = None,
        frame_reducer: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_bs: bool = True,
    ):
        """A joint CTC & Transducer ASR model.

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

        assert (
            (use_ctc and use_bs) or (use_ctc and not use_bs) or not (use_ctc and use_bs)
        ), "Blank Skip needs CTC"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

        self.use_bs = use_bs
        if self.use_bs:
            self.lconv = lconv
            self.frame_reducer = frame_reducer

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

        def forward_ctc(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            targets: List[int],
            target_lengths: torch.Tensor,
            supervisions: dict,
            subsampling_factor: int,
            ctc_beam_size: int,
            reduction: str = "sum",
            warmup: float = 1.0,
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
              supervisions:
                Dict into a pair of torch Tensor, and a list of transcription strings or token indexes
              reduction:
                Specifies the reduction to apply to the output
            """
            # Compute CTC log-prob
            ctc_output = self.ctc_output(encoder_out)  # (N, T, C)
            encoder_out_fr = encoder_out
            encoder_out_lens_fr = encoder_out_lens

            if self.use_bs and warmup >= 2.0:
                # lconv
                encoder_out = self.lconv(
                    x=encoder_out,
                    src_key_padding_mask=make_pad_mask(encoder_out_lens),
                )

                # frame reduce
                encoder_out_fr, encoder_out_lens_fr = self.frame_reducer(
                    encoder_out,
                    encoder_out_lens,
                    ctc_output,
                    target_lengths,
                    self.decoder.blank_id,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                supervision_segments, token_ids = encode_supervisions(
                    supervisions,
                    subsampling_factor=subsampling_factor,
                    token_ids=targets,
                )

            # TODO: Crash without this line
            supervision_segments = supervision_segments.to("cpu")
            decoding_graph = k2.ctc_graph(
                token_ids, modified=False, device=encoder_out.device
            )
            dense_fsa_vec = k2.DenseFsaVec(
                ctc_output,
                supervision_segments,
                allow_truncate=subsampling_factor - 1,
            )

            ctc_loss = k2.ctc_loss(
                decoding_graph=decoding_graph,
                dense_fsa_vec=dense_fsa_vec,
                output_beam=ctc_beam_size,
                reduction=reduction,
                use_double_scores=True,
            )

            return ctc_loss, encoder_out_fr, encoder_out_lens_fr

        def forward_transducer(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            y: k2.RaggedTensor,
            y_lens: torch.Tensor,
            prune_range: int = 5,
            am_scale: float = 0.0,
            lm_scale: float = 0.0,
            delay_penalty: float = 0.0,
            reduction: str = "sum",
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
              reduction:
                Specifies the reduction to apply to the output
            """
            # Now for the decoder, i.e., the prediction network
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
            boundary = torch.zeros(
                (encoder_out.size(0), 4),
                dtype=torch.int64,
                device=encoder_out.device,
            )
            boundary[:, 2] = y_lens
            boundary[:, 3] = encoder_out_lens

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
                    delay_penalty=delay_penalty,
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
                    delay_penalty=delay_penalty,
                    reduction=reduction,
                )

            return simple_loss, pruned_loss

        def forward(
            self,
            x: torch.Tensor,
            x_lens: torch.Tensor,
            y: k2.RaggedTensor,
            supervisions: dict,
            prune_range: int = 5,
            am_scale: float = 0.0,
            lm_scale: float = 0.0,
            subsampling_factor: int = 4,
            ctc_beam_size: int = 10,
            delay_penalty: float = 0.0,
            reduction: str = "sum",
            warmup: float = 1.0,
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
              reduction:
                Specifies the reduction to apply to the output
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

            assert x.size(0) == x_lens.size(0) == y.dim0

            # Compute encoder outputs
            encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

            row_splits = y.shape.row_splits(1)
            y_lens = row_splits[1:] - row_splits[:-1]

            if self.use_ctc:
                # Compute CTC loss
                ctc_loss, encoder_out, encoder_out_lens = self.forward_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=y.tolist(),
                    target_lengths=y_lens,
                    supervisions=supervisions,
                    subsampling_factor=subsampling_factor,
                    ctc_beam_size=ctc_beam_size,
                    reduction=reduction,
                    warmup=warmup,
                )
            else:
                ctc_loss = torch.empty(0, device=encoder_out.device)

            if self.use_transducer:
                # Compute transducer loss
                simple_loss, pruned_loss = self.forward_transducer(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    y=y.to(x.device),
                    y_lens=y_lens,
                    prune_range=prune_range,
                    am_scale=am_scale,
                    lm_scale=lm_scale,
                    reduction=reduction,
                    delay_penalty=delay_penalty,
                )
            else:
                simple_loss = torch.empty(0)
                pruned_loss = torch.empty(0)

            return simple_loss, pruned_loss, ctc_loss
