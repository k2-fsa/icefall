# Copyright    2022  Xiaomi Corp.        (author: Wei Kang)
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


from typing import Dict, List, Optional, Tuple

import logging
import k2
import torch
import torch.nn as nn

from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledEmbedding,
    ScaledLinear,
)
from torch.distributions.categorical import Categorical
from transformer import (
    PositionalEncoding,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    decoder_padding_mask,
    generate_square_subsequent_mask,
)
from icefall.utils import add_sos, add_eos, make_pad_mask


def _roll_by_shifts(
    src: torch.Tensor, shifts: torch.LongTensor
) -> torch.Tensor:
    """Roll tensor with different shifts for each row.
    Note:
      We assume the src is a 3 dimensions tensor and roll the last dimension.
    Example:
      >>> src = torch.arange(15).reshape((1,3,5))
      >>> src
      tensor([[[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]]])
      >>> shift = torch.tensor([[1, 2, 3]])
      >>> shift
      tensor([[1, 2, 3]])
      >>> _roll_by_shifts(src, shift)
      tensor([[[ 4,  0,  1,  2,  3],
               [ 8,  9,  5,  6,  7],
               [12, 13, 14, 10, 11]]])
    """
    assert src.dim() == 3
    (B, T, S) = src.shape
    assert shifts.shape == (B, T)

    index = (
        torch.arange(S, device=src.device)
        .view((1, S))
        .repeat((T, 1))
        .repeat((B, 1, 1))
    )
    index = (index - shifts.reshape(B, T, 1)) % S
    return torch.gather(src, 2, index)


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        quasi_joiner: nn.Module,
        transformer_lm: nn.Module,
        embedding_enhancer: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
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
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.quasi_joiner = quasi_joiner
        self.transformer_lm = transformer_lm
        self.embedding_enhancer = embedding_enhancer
        self.decoder_dim = decoder_dim

        self.simple_am_proj = ScaledLinear(
            encoder_dim, vocab_size, initial_speed=0.5
        )
        self.simple_lm_proj = ScaledLinear(decoder_dim, vocab_size)

    def get_wer(
        self,
        sampled_paths: torch.Tensor,
        y_padded: torch.Tensor,
        y_lens: torch.Tensor,
        blank_id: int = 0,
    ):
        batch_size, path_length = sampled_paths.shape
        assert y_padded.size(0) == batch_size, (y_padded.shape, batch_size)
        assert y_lens.size(0) == batch_size, (y_lens.shape, batch_size)

        device = sampled_paths.device

        px = k2.RaggedTensor(sampled_paths)
        px = px.remove_values_eq(blank_id)
        row_splits = px.shape.row_splits(1)
        px_lens = row_splits[1:] - row_splits[:-1]
        px = px.pad(mode="constant", padding_value=blank_id)

        boundary = torch.cat(
            [
                torch.zeros((batch_size, 2), dtype=torch.int64, device=device),
                px_lens.reshape(batch_size, 1),
                y_lens.reshape(batch_size, 1),
            ],
            dim=1,
        )

        # wer : (batch_size, S, U)
        wer = k2.levenshtein_distance(
            px=px, py=y_padded.int(), boundary=boundary
        )

        wer = torch.gather(
            wer,
            1,
            boundary[:, 2]
            .reshape(wer.size(0), 1, 1)
            .expand(wer.size(0), 1, wer.size(2)),
        ).squeeze(1)

        wer = torch.gather(
            wer, 1, boundary[:, 3].reshape(batch_size, 1)
        ).squeeze(1)

        # wer: (batch_size,)
        return wer

    def get_init_contexts(
        self,
        px_grad: torch.Tensor,
        py_grad: torch.Tensor,
        y_padded: torch.Tensor,
    ):
        context_size = self.decoder.context_size
        blank_id = self.decoder.blank_id
        # Get contexts for each frame according to the gradients, just like we
        # do for getting pruning bounds.
        (B, S, T1) = px_grad.shape
        T = py_grad.shape[-1]
        # shape : (B, S, T)
        tot_grad = px_grad[:, :, :T] + py_grad[:, :S, :]
        # shape : (B, T)
        best_idx = torch.argmax(tot_grad, dim=1)
        # shape : (B, T, context_size)
        state_idx = best_idx.reshape((B, T, 1)).expand(
            (B, T, context_size)
        ) + torch.arange(context_size, device=px_grad.device)
        # shape : (B, context_size)
        init_context = torch.tensor(
            [blank_id], dtype=torch.int64, device=px_grad.device
        ).expand(B, context_size)
        # shape : (B, S + context_size)
        sos_y_padded = torch.cat([init_context, y_padded], dim=1)
        init_context = torch.gather(
            sos_y_padded.unsqueeze(1).expand(B, T, S + context_size),
            dim=2,
            index=state_idx,
        )
        return init_context

    def delta_wer(
        self,
        encoder_out: torch.Tensor,
        enhanced_encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        init_context: torch.Tensor,
        y_padded: torch.Tensor,
        y_lens: torch.Tensor,
        num_pairs: int = 10,
        path_length: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          encoder_out:
            The output of the encoder whose shape is (batch_size, T, encoder_dim)
          encoder_out_lens:
            A tensor of shape (batch_size,) containing the number of frames
            before padding.
          init_context:
            A tensor of shape (batch_size, T, context_size) containing the
            initial history symbols for each frame.
          y_padded:
            The transcripts whose shape is (batch_size, S).
          y_lens:
            A tensor of shape (batch_size,) containing the number of symbols
            before padding.
          path_length:
            The length of the sampled paths.

        Returns:
          Return three tensors,
          - The delta_wer, its shape is (batch_size,)
          - The absolute value of wer_diff, its shape is (batch_size,)
          - The absolute value of pred_wer_diff, its shape is (batch_size,).
        """
        batch_size, T, encoder_dim = encoder_out.shape
        assert y_padded.size(0) == batch_size, (y_padded.shape, batch_size)
        device = encoder_out.device

        blank_id = self.decoder.blank_id
        context_size = self.decoder.context_size
        decoder_dim = self.decoder_dim
        vocab_size = self.decoder.vocab_size

        # t_index contains the frame ids we are sampling for each pair of paths.
        # shape : (batch_size, num_pairs)
        t_index = torch.arange(
            0, T + num_pairs, int(T / num_pairs), device=device
        )[0:num_pairs]
        t_index = t_index.reshape((1, num_pairs)).expand(batch_size, num_pairs)
        t_index = torch.remainder(t_index, encoder_out_lens.reshape(-1, 1))

        # The max frame index for each path
        # shape : (batch_size, num_pairs)
        t_index_max = encoder_out_lens.view(batch_size, 1, 1).expand(
            batch_size, num_pairs, 2
        )

        # left_symbols contains the left contexts of decoder for each path
        # shape : (batch_size, num_pairs, context_size)
        left_symbols = torch.gather(
            init_context,
            dim=1,
            index=t_index.unsqueeze(2).expand(
                batch_size, num_pairs, context_size
            ),
        )

        # we will sample two paths for each sequence
        t_index = t_index.reshape((batch_size, num_pairs, 1)).expand(
            batch_size, num_pairs, 2
        )
        left_symbols = left_symbols.reshape(
            (batch_size, num_pairs, 1, context_size)
        ).expand(batch_size, num_pairs, 2, context_size)

        left_symbols = left_symbols.reshape(
            batch_size * num_pairs * 2, context_size
        )

        # It has a shape of (batch_size, num_pairs) indicating whether having different
        # paths for this sequence.
        has_diff = torch.zeros((batch_size, num_pairs), device=device).bool()
        # It has a shape of (batch_size, num_pairs) indicating whether reaching final
        # for this sequence
        reach_final = torch.zeros((batch_size, num_pairs), device=device).bool()

        # The pred_wer, default zeros. If there is no different symbol for the
        # sampled paths, the pred_wers for the two paths are the same.
        pred_wer = torch.zeros((batch_size, num_pairs, 2), device=device)

        dummy_output = torch.zeros(
            (batch_size, num_pairs, vocab_size), device=device
        )

        sampled_paths_list = []
        sampled_joiner_list = []
        sampled_quasi_joiner_list = []

        while len(sampled_paths_list) < path_length:
            # (B, num_pairs, 2, encoder_dim)
            current_encoder_out = torch.gather(
                encoder_out.view(batch_size, T, 1, encoder_dim).expand(
                    batch_size, T, 2, encoder_dim
                ),
                dim=1,
                index=t_index.unsqueeze(3).expand(
                    batch_size, num_pairs, 2, encoder_dim
                ),
            )

            current_enhanced_encoder_out = torch.gather(
                enhanced_encoder_out.view(batch_size, T, 1, encoder_dim).expand(
                    batch_size, T, 2, encoder_dim
                ),
                dim=1,
                index=t_index.unsqueeze(3).expand(
                    batch_size, num_pairs, 2, encoder_dim
                ),
            )

            # (B, num_pairs, 2, decoder_dim)
            decoder_output = self.decoder(left_symbols, need_pad=False).view(
                batch_size, num_pairs, 2, decoder_dim
            )
            # joiner_output : (B, num_pairs, 2, V);
            joiner_output = self.joiner(current_encoder_out, decoder_output)
            # quasi_joiner_output : (B, num_pairs, 2, V)
            quasi_joiner_output = self.quasi_joiner(
                current_enhanced_encoder_out, decoder_output
            )

            probs = torch.softmax(joiner_output, -1)
            # sampler: https://pytorch.org/docs/stable/distributions.html#categorical
            sampler = Categorical(probs=probs)

            # sample one symbol for each path
            # index : (batch_size, num_pairs, 2)
            index = sampler.sample()

            # The two paths have different symbols.
            mask = index[:, :, 0] != index[:, :, 1]

            # shape : (batch_size, num_pairs), will only be True when the two paths have
            # different symbols in the first time.
            meet_diff = mask & ~has_diff & ~reach_final

            has_diff |= mask

            # wer_output: (B, num_pairs, 2)
            wer_output = torch.gather(
                quasi_joiner_output, dim=3, index=index.unsqueeze(3)
            ).squeeze(3)

            # we only get the pred_wer at the position where the two paths start
            # to have different symbols.
            pred_wer = torch.where(
                meet_diff.reshape(batch_size, num_pairs, 1),
                wer_output,
                pred_wer,
            )

            # update (t, s) for each path
            # index == 0 means the sampled symbol is blank
            # t_mask : (B, num_pairs, 2)
            t_mask = index == 0
            t_index = t_index + 1

            # if reaching final, we will ignore the sampled symbols, just append
            # blank_id to the paths.
            index = torch.where(
                reach_final.reshape(batch_size, num_pairs, 1).expand(
                    batch_size, num_pairs, 2
                ),
                blank_id,
                index,
            )
            sampled_paths_list.append(index)

            joiner_output = torch.where(
                reach_final.reshape(batch_size, num_pairs, 1).expand(
                    batch_size, num_pairs, vocab_size
                ),
                dummy_output,
                joiner_output[:, :, 0, :],
            )
            sampled_joiner_list.append(joiner_output)

            quasi_joiner_output = torch.where(
                reach_final.reshape(batch_size, num_pairs, 1).expand(
                    batch_size, num_pairs, vocab_size
                ),
                dummy_output,
                quasi_joiner_output[:, :, 0, :],
            )
            sampled_quasi_joiner_list.append(quasi_joiner_output)

            final_mask = t_index >= t_index_max

            # Set reach_final to true when one of the paths reaching final.
            reach_final = (
                reach_final | final_mask[:, :, 0] | final_mask[:, :, 1]
            )

            t_index.masked_fill_(final_mask, 0)

            left_symbols = left_symbols.view(
                batch_size, num_pairs, 2, context_size
            )
            current_symbols = torch.cat(
                [
                    left_symbols,
                    index.unsqueeze(3),
                ],
                dim=3,
            )
            # if the sampled symbol is blank, we only need to roll the history
            # symbols, if the sampled symbol is not blank, append the newly
            # sampled symbol.
            left_symbols = _roll_by_shifts(
                current_symbols.view(
                    batch_size, num_pairs * 2, context_size + 1
                ),
                t_mask.view(batch_size, num_pairs * 2).to(torch.int64),
            )
            left_symbols = left_symbols[:, :, 1:]

            left_symbols = left_symbols.view(
                batch_size * num_pairs * 2, context_size
            )

        # sampled_paths : (batch_size, num_pairs, 2, path_lengths)
        sampled_paths = torch.stack(sampled_paths_list, dim=3).int()

        # sampled_joiner : (batch_size, num_pairs, path_lengths, vocab_size)
        sampled_joiner = torch.stack(sampled_joiner_list, dim=2)
        sampled_quasi_joiner = torch.stack(sampled_quasi_joiner_list, dim=2)

        y_padded_expand = (
            y_padded.reshape(y_padded.size(0), 1, y_padded.size(1))
            .expand(y_padded.size(0), num_pairs, y_padded.size(1))
            .reshape(y_padded.size(0) * num_pairs, y_padded.size(1))
        )
        y_expand_lens = (
            y_lens.reshape(y_lens.size(0), 1)
            .expand(y_lens.size(0), num_pairs)
            .reshape(
                y_lens.size(0) * num_pairs,
            )
        )

        wer1 = self.get_wer(
            sampled_paths=sampled_paths[:, :, 0, :].view(
                batch_size * num_pairs, path_length
            ),
            y_padded=y_padded_expand,
            y_lens=y_expand_lens,
            blank_id=blank_id,
        )
        wer1 = wer1.view(batch_size, num_pairs)

        wer2 = self.get_wer(
            sampled_paths=sampled_paths[:, :, 1, :].view(
                batch_size * num_pairs, path_length
            ),
            y_padded=y_padded_expand,
            y_lens=y_expand_lens,
            blank_id=blank_id,
        )
        wer2 = wer2.view(batch_size, num_pairs)

        wer_diff = wer1 - wer2
        pred_wer_diff = pred_wer[:, :, 0] - pred_wer[:, :, 1]

        return (wer_diff, pred_wer_diff, sampled_joiner, sampled_quasi_joiner)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        sos_id: int,
        eos_id: int,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        num_pairs: int = 10,
        path_length: int = 20,
        warmup: float = 1.0,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert reduction in ("sum", "none"), reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        context_size = self.decoder.context_size
        vocab_size = self.decoder.vocab_size
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
            (x.size(0), 4), dtype=torch.int64, device=x.device
        )
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
                rnnt_type="constrained",
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
                rnnt_type="constrained",
                reduction=reduction,
            )

        text_embedding, text_embedding_key_padding_mask = self.transformer_lm(
            y, sos_id=sos_id, eos_id=eos_id
        )

        embedding_key_padding_mask = make_pad_mask(x_lens)
        enhanced_embedding = self.embedding_enhancer(
            embedding=encoder_out.detach(),
            text_embedding=text_embedding,
            embedding_key_padding_mask=embedding_key_padding_mask,
            text_embedding_key_padding_mask=text_embedding_key_padding_mask,
            warmup=warmup,
        )

        l2_loss = torch.sum(
            torch.pow(enhanced_embedding - encoder_out.detach(), 2)
        ) / encoder_out.size(2)

        init_context = self.get_init_contexts(
            px_grad=px_grad, py_grad=py_grad, y_padded=y_padded
        )

        # wer_diff, pred_wer_diff : (B, num_pairs)
        (
            wer_diff,
            pred_wer_diff,
            sampled_joiner,
            sampled_quasi_joiner,
        ) = self.delta_wer(
            encoder_out=encoder_out,
            enhanced_encoder_out=enhanced_embedding.detach(),
            encoder_out_lens=x_lens,
            init_context=init_context,
            y_padded=y_padded,
            y_lens=y_lens,
            num_pairs=num_pairs,
            path_length=path_length,
        )

        delta_wer = torch.pow(wer_diff - pred_wer_diff, 2)

        delta_wer_loss = torch.sum(delta_wer)

        predictor_wer_loss = torch.sum(
            sampled_joiner * sampled_quasi_joiner.detach()
        ) / (num_pairs * path_length * sampled_joiner.size(-1))

        return (
            simple_loss,
            pruned_loss,
            delta_wer_loss,
            l2_loss,
            predictor_wer_loss,
        )


class TransformerLM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ) -> None:
        """
        Args:
          num_classes:
            The output dimension of the model.
          d_model:
            Attention dimension.
          nhead:
            Number of heads in multi-head attention.
            Must satisfy d_model // nhead == 0.
          dim_feedforward:
            The output dimension of the feedforward layers in encoder/decoder.
          num_encoder_layers:
            Number of encoder layers.
          dropout:
            Dropout in encoder/decoder.
          layer_dropout (float): layer-dropout rate.
        """
        super().__init__()

        self.num_classes = num_classes

        self.encoder_pos = PositionalEncoding(d_model, dropout)

        self.embed = ScaledEmbedding(
            num_embeddings=self.num_classes, embedding_dim=d_model
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.encoder_output_layer = ScaledLinear(
            d_model, num_classes, bias=True
        )

    def forward(
        self,
        y: k2.RaggedTensor,
        sos_id: int,
        eos_id: int,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          Return a tuple containing 2 tensors:
            - Encoder output with shape (T, N, C). It can be used as key and
              value for the decoder.
            - Encoder output padding mask. It can be used as
              memory_key_padding_mask for the decoder. Its shape is (N, T).
        """
        assert y.num_axes == 2, y.num_axes

        device = y.device

        sos_y = add_sos(y, sos_id=sos_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=sos_id)
        y_eos = add_eos(y, eos_id=eos_id)
        y_eos_padded = y_eos.pad(mode="constant", padding_value=eos_id)

        att_mask = generate_square_subsequent_mask(y_eos_padded.shape[-1]).to(
            device
        )

        key_padding_mask = decoder_padding_mask(y_eos_padded, ignore_id=eos_id)
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        key_padding_mask[:, 0] = False

        x = self.embed(sos_y_padded)
        x = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        x = self.encoder(
            x,
            mask=att_mask,
            src_key_padding_mask=key_padding_mask,
            warmup=warmup,
        )  # (T, N, C)

        return x, key_padding_mask


class EmbeddingEnhancer(nn.Module):
    """
    Enhance the encoder embedding to "knows about" the text as well as the acoustics.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ):
        super().__init__()
        self.encoder_pos = PositionalEncoding(d_model, dropout)
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )
        self.enhancer = TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        embedding_mask: Optional[torch.Tensor] = None,
        text_embedding_mask: Optional[torch.Tensor] = None,
        embedding_key_padding_mask: Optional[torch.Tensor] = None,
        text_embedding_key_padding_mask: Optional[torch.Tensor] = None,
        mask_proportion: float = 0.25,
        warmup: float = 1.0,
    ):
        N, T, C = embedding.shape
        mask = torch.randn((N, T, C), device=embedding.device)
        mask = mask > mask_proportion
        masked_embedding = torch.masked_fill(embedding, ~mask, 0.0)
        masked_embedding = self.encoder_pos(masked_embedding)
        masked_embedding = masked_embedding.permute(1, 0, 2)

        enhanced_embedding = self.enhancer(
            tgt=masked_embedding,
            memory=text_embedding,
            tgt_mask=embedding_mask,
            memory_mask=embedding_mask,
            tgt_key_padding_mask=embedding_key_padding_mask,
            memory_key_padding_mask=text_embedding_key_padding_mask,
            warmup=warmup,
        )

        # shape: (N, T, C)
        enhanced_embedding = enhanced_embedding.permute(1, 0, 2)
        return enhanced_embedding
