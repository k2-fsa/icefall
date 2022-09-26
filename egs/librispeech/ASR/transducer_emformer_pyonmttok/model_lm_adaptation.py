# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, C) and (N, U, C). Its
            output shape is (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert hasattr(decoder, "blank_id")

        self.criterion_lm = nn.NLLLoss(ignore_index=-1, reduction="sum")
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> torch.Tensor:
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

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, C]
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

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=decoder_out.float(),
                am=encoder_out.float(),
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
        # am_pruned : [B, T, prune_range, C]
        # lm_pruned : [B, T, prune_range, C]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=encoder_out, lm=decoder_out, ranges=ranges
        )

        # logits : [B, T, prune_range, C]
        logits = self.joiner(am_pruned, lm_pruned)

        logits_lm = self.joiner.forward_lm(decoder_out)[:, :-1, :].contiguous()
        loss_lm = self.criterion_lm(
            logits_lm.view(-1, logits_lm.size(-1)), y_padded.view(-1) - 1
        )

        # [B, S + 1, C]
        # comment that
        # ((1-p)*torch.eye(decoder_out.size(-1)) + p * self.joiner_fixed.forward_lm(decoder_out_fixed) ) * softmax(self.joiner.forward_lm(decoder_out))
        # print(
        #     logits.shape,
        #     am_pruned.shape,
        #     lm_pruned.shape,
        #     encoder_out.shape,
        #     decoder_out.shape,
        # )
        # print(torch.argmax(logits[-1, :, 0, :], dim=-1))
        # print(1 + torch.argmax(logits[-1, :, 0, 1:], dim=-1))
        # print(torch.argmax(logits[-1, :10, 1, :], dim=-1))
        # print(y_padded.shape)
        # print(y_padded[-1, :])
        # print("end")
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss, loss_lm)


import copy


class TransducerStep2(nn.Module):
    def __init__(self, decoder, joiner):
        super().__init__()
        self.transducer = Transducer(None, decoder, joiner)
        # self.transducer_fixed = type(self.transducer)(
        #     None, decoder, joiner
        # )  # get a new instance
        # self.transducer_fixed.load_state_dict(self.transducer.state_dict())
        sd_ref = copy.deepcopy(self.transducer.state_dict())
        self.transducer_fixed = copy.deepcopy(self.transducer)
        self.transducer_fixed.load_state_dict(sd_ref)
        self.transducer_fixed.eval()
        for param in self.transducer_fixed.parameters():
            param.requires_grad = False
        for param in self.transducer.decoder.parameters():
            param.requires_grad = False
        for param in self.transducer.joiner.parameters():
            param.requires_grad = True
        self.criterion_lm = nn.NLLLoss(ignore_index=-1, reduction="none")
        self.p = 0.5

    @classmethod
    def compute_loss(self, p_fixed, logp_train, y_padded, p):
        # y_padded_zero = y_padded - 1
        # y_padded_zero[y_padded_zero < 0] = 0
        # print(y_padded)
        y_padded_zero = y_padded
        index = torch.nn.functional.one_hot(
            y_padded_zero.to(torch.int64), p_fixed.size(-1)
        )
        eye = torch.zeros_like(p_fixed)
        eye = eye.scatter_(
            dim=-1, index=y_padded_zero[:, :, None].long(), value=1
        )
        eye[(y_padded_zero == 0)[:, :, None].repeat(1, 1, eye.size(-1))] = 0
        ilm = eye * logp_train
        kl = p_fixed * logp_train  # - p_fixed * torch.log(p_fixed)
        # prob_blank = p * torch.ones(
        #     (1, 1, p_fixed.size(-1)), device=p_fixed.device
        # )
        # prob_blank[0, 0, 0] = 10000
        loss = ((1 - p) * eye + p * p_fixed) * logp_train
        # return ((1 - p) * ilm + p * kl), ilm, kl
        # print(kl.sum(), loss.sum())
        return loss, (1 - p) * ilm, p * kl

    @classmethod
    def compute_loss_no_blank(self, p_fixed, logp_train, y_padded, p):
        y_padded_zero = y_padded - 1
        y_padded_zero[y_padded_zero < 0] = 0
        # print(y_padded)
        index = torch.nn.functional.one_hot(
            y_padded_zero.to(torch.int64), p_fixed.size(-1)
        )
        eye = torch.zeros_like(p_fixed)
        eye = eye.scatter_(
            dim=-1, index=y_padded_zero[:, :, None].long(), value=1
        )
        eye[(y_padded_zero == 0)[:, :, None].repeat(1, 1, eye.size(-1))] = 0
        ilm = eye * logp_train
        kl = p_fixed * logp_train - p_fixed * torch.log(p_fixed)
        # prob_blank = p * torch.ones(
        #     (1, 1, p_fixed.size(-1)), device=p_fixed.device
        # )
        # prob_blank[0, 0, 0] = 10000
        loss = (
            (1 - p) * eye + p * p_fixed
        ) * logp_train - p * p_fixed * torch.log(p_fixed)
        # return ((1 - p) * ilm + p * kl), ilm, kl
        # print(kl.sum(), loss.sum())
        return loss, (1 - p) * ilm, p * kl

    @classmethod
    def compute_loss_no_blank_paper(self, p_fixed, logp_train, y_padded, p):
        p = 0.5
        y_padded_zero = y_padded - 1
        y_padded_zero[y_padded_zero < 0] = 0
        # print(y_padded)
        index = torch.nn.functional.one_hot(
            y_padded_zero.to(torch.int64), p_fixed.size(-1)
        )
        eye = torch.zeros_like(p_fixed)
        eye = eye.scatter_(
            dim=-1, index=y_padded_zero[:, :, None].long(), value=1
        )
        eye[(y_padded_zero == 0)[:, :, None].repeat(1, 1, eye.size(-1))] = 0
        ilm = eye * logp_train
        kl = p_fixed * logp_train  # - p_fixed * torch.log(p_fixed)
        # prob_blank = p * torch.ones(
        #     (1, 1, p_fixed.size(-1)), device=p_fixed.device
        # )
        # prob_blank[0, 0, 0] = 10000
        loss = ((1 - p) * eye + p * p_fixed) * logp_train
        # return ((1 - p) * ilm + p * kl), ilm, kl
        # print(kl.sum(), loss.sum())
        return loss, (1 - p) * ilm, p * kl

    def compute_loss_no_blank_paper_func(
        self, p_fixed_full, logp_train, logp_train_full, y_padded, p
    ):
        p = 0.9
        y_padded_zero = y_padded - 1
        ilm = -self.criterion_lm(
            logp_train.contiguous().view(-1, logp_train.size(-1)),
            y_padded_zero.view(-1).to(torch.int64),
        ).sum()
        kl = (p_fixed_full * logp_train_full).sum(dim=-1).view(-1).sum()
        return (1 - p) * ilm + p * kl,  ilm, kl

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> torch.Tensor:
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
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """

        blank_id = self.transducer.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        y_padded = y.pad(mode="constant", padding_value=0)
        # decoder_out: [B, S + 1, C]
        decoder_out = self.transducer.decoder(sos_y_padded)

        with torch.no_grad():
            self.transducer_fixed.eval()
            decoder_out_fixed = self.transducer_fixed.decoder(sos_y_padded)
            raw_output_fixed, full_raw_output_fixed = self.transducer_fixed.joiner.forward_lm_raw(
                decoder_out_fixed
            )
            p_fixed_full = torch.exp(
                full_raw_output_fixed[:, :, :].log_softmax(dim=-1)
            ).contiguous()
            p_blank_fixed = p_fixed_full[:,:,0]
            p_fixed = torch.exp(
                raw_output_fixed[:, :, :].log_softmax(dim=-1)
            ).contiguous()
            sum_raw_output_fixed = raw_output_fixed.sum(dim=-1)

        raw_output, full_raw_output = self.transducer.joiner.forward_lm_raw(decoder_out_fixed)
        logp_train = raw_output[:, :, :].log_softmax(dim=-1).contiguous()
        logp_train_full = full_raw_output[:, :, :].log_softmax(dim=-1).contiguous()
        logp_blank_train = logp_train_full[:,:,0]

        # print(p_fixed.shape)
        # print(torch.max(p_fixed[0, :10], dim=-1))
        # print(torch.max(torch.exp(logp_train)[0, :10], dim=-1))
        # print("y", sos_y_padded[0, :10])
        # sum_raw_output = raw_output.sum(dim=-1)
        # self.transducer_fixed.eval()
        # self.transducer.eval()
        # print(
        #     torch.max(
        #         self.transducer.joiner.output_linear._parameters["weight"]
        #         - self.transducer_fixed.joiner.output_linear._parameters[
        #             "weight"
        #         ]
        #     ),
        #     torch.min(
        #         self.transducer.joiner.output_linear._parameters["weight"]
        #         - self.transducer_fixed.joiner.output_linear._parameters[
        #             "weight"
        #         ]
        #     ),
        # )
        # print(
        #     torch.max(
        #         self.transducer.joiner.output_linear._parameters["weight"][0]
        #         - self.transducer_fixed.joiner.output_linear._parameters[
        #             "weight"
        #         ][0]
        #     ),
        #     torch.min(
        #         self.transducer.joiner.output_linear._parameters["weight"][0]
        #         - self.transducer_fixed.joiner.output_linear._parameters[
        #             "weight"
        #         ][0]
        #     ),
        # )
        # print(
        #     torch.max(
        #         self.transducer.joiner.output_linear._parameters["weight"][1]
        #         - self.transducer_fixed.joiner.output_linear._parameters[
        #             "weight"
        #         ][1]
        #     ),
        #     torch.min(
        #         self.transducer.joiner.output_linear._parameters["weight"][1]
        #         - self.transducer_fixed.joiner.output_linear._parameters[
        #             "weight"
        #         ][1]
        #     ),
        # )
        # # print(
        # #     torch.max(
        # #         self.transducer.joiner.forward_lm_full(decoder_out)[:, :, 0]
        # #         - self.transducer_fixed.joiner.forward_lm_full(decoder_out)[
        # #             :, :, 0
        # #         ]
        # #     )
        # # )
        # # print(
        # #     torch.max(torch.exp(
        # #         self.transducer.joiner.forward_lm_full(decoder_out)
        # #     )[:,:,0]
        # #     - torch.exp(
        # #         self.transducer_fixed.joiner.forward_lm_full(decoder_out))[
        # #             :, :, 0
        # #         ]
        # #     )
        # # )
        # print("output sum", ((sum_raw_output-sum_raw_output_fixed)**2).sum())
        # self.transducer_fixed.train()
        # self.transducer.train()
        # eye[i, j , y_padded_zero[i, j]] = 1
        # print(p_fixed.shape, logp_train.shape, y_padded.shape)
        loss, ilm, kl = self.compute_loss_no_blank_paper_func(
            p_fixed_full, logp_train[:, :-1, :], logp_train_full, y_padded, self.p
        )
        # norm_loss = ((sum_raw_output-sum_raw_output_fixed)**2).sum()
        # + 5*norm_loss / sos_y_padded.size(0)
        # -10*(logp_blank_train*p_blank_fixed).sum()
        return -loss.sum()-5*(logp_blank_train*p_blank_fixed).sum(), -(logp_blank_train*p_blank_fixed).sum(), -kl.sum()


class TransducerStep2Hybrid(TransducerStep2):
    def __init__(self, decoder, joiner, transformer_decoder):
        super().__init__(decoder, joiner)
        self.transducer.decoder = transformer_decoder
