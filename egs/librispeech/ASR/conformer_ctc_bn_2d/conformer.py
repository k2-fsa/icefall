#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
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

import copy
import random
import math
import warnings
from typing import Optional, Tuple, List
import torch_flow_sampling

import torch
from torch import Tensor, nn
from subsampling import Conv2dSubsampling, VggSubsampling
from transformer import Supervisions, TransformerEncoderLayer, TransformerDecoderLayer, encoder_padding_mask, \
        LabelSmoothingLoss, PositionalEncoding, pad_sequence, add_sos, add_eos, decoder_padding_mask, \
        generate_square_subsequent_mask


class ConformerTrunk(nn.Module):
    def __init__(self,
                 num_features: int,
                 subsampling_factor: int = 4,
                 d_model: int = 256,
                 nhead: int = 4,
                 dim_feedforward: int = 2048,
                 num_layers: int = 10,
                 dropout: float = 0.1,
                 cnn_module_kernel: int = 31,
                 use_feat_batchnorm: bool = True) -> None:
        super(ConformerTrunk, self).__init__()
        if use_feat_batchnorm:
            self.feat_batchnorm = nn.BatchNorm1d(num_features)

        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.feat_embed converts the input of shape [N, T, num_classes]
        # to the shape [N, T//subsampling_factor, d_model].
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_classes -> d_model
        self.feat_embed = VggSubsampling(num_features, d_model)

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = ConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            cnn_module_kernel,
        )

        self.encoder = ConformerEncoder(encoder_layer, num_layers)

    def forward(
        self, x: torch.Tensor, supervision: Optional[Supervisions] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is [N, T, C].
          supervision:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling)

        Return (output, pos_emb, mask), where:
          output:  The output embedding, of shape (T, N, C).
          pos_emb:  The positional embedding (this will be used by ctc_encoder forward).
          mask: The output padding mask, a Tensor of bool, of shape [N, T].
              It is None if `supervision` is None.
        """
        if hasattr(self, 'feat_batchnorm'):
            x = x.permute(0, 2, 1)  # [N, T, C] -> [N, C, T]
            x = self.feat_batchnorm(x)
            x = x.permute(0, 2, 1)  # [N, C, T] -> [N, T, C]

        x = self.feat_embed(x)

        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        mask = encoder_padding_mask(x.size(0), supervision)

        mask = mask.to(x.device) if mask is not None else None
        x = self.encoder(x, pos_emb=pos_emb, key_padding_mask=mask)  # (T, N, C)

        return x, pos_emb, mask


class BidirectionalConformer(nn.Module):
    """
    This is a modified conformer where the encoder outputs the probabilities of
    hidden discrete symbols.  These probabilities are sampled from and then
    given as input to two separate "forward" decoders: the CTC decoder and the
    attention decoder.  We also have a reverse attention decoder where we
    predict these discrete symbols from the word/phone/word-piece sequences.
    From that we subtract the log-probs of the symbols given a simple "self-decoder"
    that predicts those symbols given previous symbols; this avoids the
    symbols converging on the most likely ones as a result of the reverse
    attention decoder's contribution to the loss function.

    Caution: this code has several different 'forward' functions: the
    regular 'forward' function is for the encoder, but there are others:

     forward(),            [generates sampled symbols from the conformer encoder,
                            also returning some additional things, e.g. the
                            pre-sampling symbol probabilities.]
     decoder_forward(),    [predicts word-piece symbols from hidden symbols, returns
                            scalar total log-like]
     ctc_encoder_forward() [caution: this just gives the loglikes,
                            it does not do CTC decoding]
     reverse_decoder_forward() [predicts hidden symbols from word-piece
                                symbols]
     self_prediction_forward() [predicts hidden symbols from previous hidden
                             symbols using a simple model, returns scalar
                             total log-like.  We subtract this from the
                             loss function as a mechanism to avoid
                             "trivial" solutions; this can also be justified
                             from a point of view of maximizing mutual information].

    Args:
          num_features: Input acoustic feature dimension, e.g. 40.
          num_classes: Output dimension which might be number of phones or
                     word-piece symbols, including blank/sos/eos.
          subsampling_factor: Factor by which acoustic features are
                     downsampled in encoder.
          d_model:  Dimension for attention computations
          nhead:  Number of heads in attention computations
          dim_feedforward:  Dimension for feedforward computations in
                  conformer
          num_encoder_layers:  Number of encoder layers in the "trunk" that
                  encodes the acoustic features
          num_ctc_encoder_layers:  Number of layers in the CTC encoder
                    that comes after the trunk.
                    These are just conformer encoder layers.
          num_decoder_layers:  Number of layers in the attention decoder;
                    this goes from the trunk to the word-pieces or phones.
          num_reverse_encoder_layers: Number of layers in the reverse
                    encoder, which encodes the word-pieces or phones
                    and whose output will be used to predict the
                    discrete bottleneck.
          num_reverse_decoder_layers: Number of layers in the reverse
                    encoder, which predicts the discrete-bottleneck
                    samples from the word sequence.
          num_self_predictor_layers:  Number of layers in the simple
                    self-predictor model that predicts the discrete-bottleneck
                    samples from its own previous frames.  This is
                    intended to be a relatively simple model because its main
                    useful function is to prevent "trivial" solutions
                    such as collapse of the distribution to a single symbol,
                    or symbols that are highly correlated across time.
          dropout:  Dropout probability
          cnn_module_kernel: Kernel size in forward conformer layers
          is_bpe:   If false, we'll add one (for EOS) to the number of
                    classes at the output of the decoder
          use_feat_batchnorm: If true, apply batchnorm to the input features.
          discretization_tot_classes: Total number of classes
                    (across all groups) in the discrete bottleneck
          discretization_num_groups: Number of groups of classes/symbols
                   in the discrete bottleneck
    """
    def __init__(
            self,
            num_features: int,
            num_classes: int,
            subsampling_factor: int = 4,
            d_model: int = 256,
            nhead: int = 4,
            dim_feedforward: int = 2048,
            num_trunk_encoder_layers: int = 12,
            num_ctc_encoder_layers: int = 2,
            num_decoder_layers: int = 6,
            num_reverse_encoder_layers: int = 4,
            num_reverse_decoder_layers: int = 4,
            num_self_predictor_layers: int = 2,
            dropout: float = 0.1,
            cnn_module_kernel: int = 31,
            is_bpe: bool = False,
            use_feat_batchnorm: bool = True,
            discretization_tot_classes: int = 512,
            discretization_num_groups: int = 4
    ) -> None:
        super(BidirectionalConformer, self).__init__()

        self.trunk = ConformerTrunk(num_features, subsampling_factor,
                                    d_model, nhead, dim_feedforward,
                                    num_trunk_encoder_layers, dropout,
                                    cnn_module_kernel,
                                    use_feat_batchnorm)

        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor

        encoder_layer = ConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            cnn_module_kernel,
        )
        self.ctc_encoder = ConformerEncoder(encoder_layer, num_ctc_encoder_layers)
        self.ctc_output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(d_model, num_classes)
        )

        # absolute position encoding, used by various layer types
        self.abs_pos = PositionalEncoding(d_model, dropout)

        if num_decoder_layers > 0:
            # extra class for sos/eos symbol, if not BPE
            self.decoder_num_class = self.num_classes if is_bpe else self.num_classes + 1

            # self.embed is the token embedding (embedding for phones or
            # word-pieces) that is used for both the forward and reverse decoders
            self.token_embed_scale = d_model ** 0.5
            self.token_embed = nn.Embedding(
                num_embeddings=self.decoder_num_class, embedding_dim=d_model,
                _weight=torch.randn(self.decoder_num_class, d_model) * (1 / self.token_embed_scale)
            )

            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
                norm=nn.LayerNorm(d_model)
            )

            self.decoder_output_layer = torch.nn.Linear(
                d_model, self.decoder_num_class
            )

            # Caution: it takes padding_idx=-1 as a default.  That's the
            # target value it will ignore.
            self.decoder_criterion = LabelSmoothingLoss(self.decoder_num_class)
        else:
            self.decoder_criterion = None


        if num_reverse_encoder_layers > 0:
            self.reverse_encoder_pos = PositionalEncoding(d_model, dropout)

            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.reverse_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=num_reverse_encoder_layers,
                                                         norm=nn.LayerNorm(d_model))

        if num_reverse_decoder_layers > 0:

            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.reverse_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=num_reverse_encoder_layers,
                                                         norm=nn.LayerNorm(d_model))


        if num_reverse_decoder_layers > 0:

            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.reverse_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_reverse_decoder_layers,
                norm=nn.LayerNorm(d_model)
            )
            # There is no "linear output" for the reverse decoder;
            # that is handled by the discrete_bottleneck layer itself.
            # It just accepts the output of self.reverse_decoder as
            # the input to its prediction mechanism.

        if num_self_predictor_layers > 0:
            encoder_layer = SimpleCausalEncoderLayer(d_model,
                                                     dropout=dropout)
            final_linear = nn.Linear(d_model, d_model, bias=False)
            self.self_predictor_encoder = nn.Sequential(*[copy.deepcopy(encoder_layer)
                                                          for _ in range(num_self_predictor_layers)],
                                                        final_linear,
                                                        FastOffsetLayer(d_model))


        self.sample_and_predict = SampleAndPredict(
            dim=d_model,
            tot_classes=discretization_tot_classes,
            num_groups=discretization_num_groups)



    def forward(self, x: Tensor, supervision: Optional[Supervisions] = None,
                need_softmax: bool = True) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward function that "encodes" the acoustic features through the "trunk"
        (the shared part of the encoding of the encoding of the acoustic features)

        Args:
          x:
            The input tensor. Its shape is [N, T, F], i.e. [batch_size, num_frames, num_features].
        supervision:
            Supervision in lhotse format (optional; needed only for acoustic length
            information)
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling).  Used only to compute masking information.

       Returns: (memory, pos_emb, key_padding_mask), where:

           memory: a Tensor of shape [T, N, E] i.e. [T, batch_size, embedding_dim] where T
                   is actually a subsampled form of the num_frames of the input `x`.
                   If self.bypass_bottleneck, it will be taken before the discrete
                   bottleneck; otherwise, from after.
            pos_emb: The relative positional embedding; will be given to ctc_encoder_forward()
      key_padding_mask:  The padding mask for the "memory" output, a Tensor of bool of
                    shape [N, T] (only if supervision was supplied, else None).
        """
        memory, pos_emb, memory_key_padding_mask = self.trunk(x, supervision)
        return memory, pos_emb, memory_key_padding_mask


    def sample_forward(self, memory: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Given the "memory" from forward(), run the sample_and_redict module.
        See documentation for forward() of class SampleAndPredict for more info.

        Returns (sampled, softmax, positive_embed_shifted, negative_embed_shifted),
        where positive_embed_shifted, for instance, is positive_embed
        shifted by one so that positive_embed_shifted[t] == positive_embed[t-1], as in:
             (T, N, E) = positive_embed.shape
        positive_embed_shifted = torch.cat((torch.zeros(1, N, E), positive_embed[:-1,:,:]), dim=0)

        """
        (sampled, softmax, positive_embed, negative_embed) = self.sample_and_predict(memory)

        (T, N, E) = memory.shape
        device = memory.device
        zeros = torch.zeros(1, N, E).to(memory.device)
        positive_embed_shifted = torch.cat((zeros, positive_embed[:-1,:,:]), dim=0)
        negative_embed_shifted = torch.cat((zeros, negative_embed[:-1,:,:]), dim=0)

        return (sampled, softmax, positive_embed_shifted, negative_embed_shifted)

    def decoder_forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[List[int]],
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        Compute the decoder loss function (given a particular list of hypotheses).

        Args:
          memory:
            The first output of forward(), with shape [T, N, E]
          memory_key_padding_mask:
            The padding mask from forward(), a tensor of bool with shape [N, T]
          token_ids:
            A list-of-list IDs. Each sublist contains IDs for an utterance.
            The IDs can be either phone IDs or word piece IDs.
          sos_id:
            sos token id
          eos_id:
            eos token id

        Returns:
            A scalar, the **sum** of label smoothing loss over utterances
            in the batch without any normalization.
        """

        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(ys_in, batch_first=True, padding_value=eos_id)

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(ys_out, batch_first=True, padding_value=-1)

        device = memory.device
        ys_in_pad = ys_in_pad.to(device)
        ys_out_pad = ys_out_pad.to(device)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.token_embed(ys_in_pad) * self.token_embed_scale # (N, T) -> (N, T, C)
        tgt = self.abs_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # ,tgt_key_padding_mask=tgt_key_padding_mask,...
        # We don't supply tgt_key_padding_mask because it's useless; thanks to tgt_mask,
        # those positions would already be excluded from consideration for any output
        # position that we're going to care about.
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (T, N, C)

        pred_pad = pred_pad.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        pred_pad = self.decoder_output_layer(pred_pad)  # (N, T, C)

        decoder_loss = self.decoder_criterion(pred_pad, ys_out_pad)

        return decoder_loss

    def ctc_encoder_forward(
        self,
        memory: torch.Tensor,
        pos_emb: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Passes the output of forward() through the CTC encoder and the CTC
        output to give the output that can be given to the CTC loss function

        Args:
          memory:
            It's the output of forward(), with shape (T, N, E)
          pos_emb:
             Relative positional embedding tensor: (N, 2*T-1, E)
          memory_key_padding_mask:
            The padding mask from forward(), a tensor of bool of shape (N, T)

        Returns:
            A Tensor with shape [N, T, C] where C is the number of classes
            (e.g. number of phones or word pieces).  Contains normalized
            log-probabilities.
        """
        x = self.ctc_encoder(memory,
                             pos_emb,
                             key_padding_mask=memory_key_padding_mask)
        x = self.ctc_output_layer(x)
        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        x = nn.functional.log_softmax(x, dim=-1)  # (N, T, C)
        return x


    def self_prediction_forward(
            self,
            negative_embed_shifted: torch.Tensor,
            memory_key_padding_mask: torch.Tensor,
            sampled: torch.Tensor,
            softmax: Optional[torch.Tensor]) -> Tensor:
        """
        Returns the total log-prob of the the labels sampled in the discrete
        bottleneck layer, as predicted using a relatively simple model that
        predicts from previous frames sampled from the bottleneck layer.
        [Appears on the denominator of an expression for mutual information].

        Args:
          negative_embed_shifted:
            The negative_embed_shifted output of self.sample_forward(), with shape [T, N, E]
          memory_key_padding_mask:
            The padding mask from the encoder, of shape [N, T], boolean, True
            for masked locations.
          sampled: sampled and interpolated one-hot values, as a Tensor of shape [T, N, C]
              where C corresponds to `discretization_tot_classes`
              as given to the constructor.  This will be needed for the 'reverse'
              model.
          softmax: is a "soft" version of `sampled`; if None, will default to `sampled`.

        Returns:
            A scalar tensor, the **sum** of the log-prob loss over utterances
            in the batch without any normalization.
        """
        # no mask is needed for self_predictor_encoder; its CNN
        # layer uses left-padding only, making it causal, so the mask
        # is redundant (it wouldn't affect any of the
        # outputs we care about).
        predictor = self.self_predictor_encoder(negative_embed_shifted)

        prob = self.sample_and_predict.compute_prob(predictor,
                                                    sampled, softmax,
                                                    memory_key_padding_mask,
                                                    reverse_grad=True)
        return prob


    def reverse_decoder_forward(
            self,
            positive_embed_shifted: torch.Tensor,
            memory_key_padding_mask: torch.Tensor,
            sampled: torch.Tensor,
            softmax: Optional[torch.Tensor],
            token_ids: List[List[int]],
            sos_id: int,
            eos_id: int,
            padding_id: int,
    ) -> torch.Tensor:
        """
        This is the reverse decoder function, which returns the total probability of the
        labels sampled in the discrete bottleneck layer, as predicted from the
        supervision word-sequence.  Caution: it has the opposite sign from
        the result of decoder_forward().

        Args:
          positive_embed_shifted:
            It's the positive_embed_shifted output of self.sample_forward(), with
            shape [T, N, E]
          memory_key_padding_mask:
            The padding mask from the encoder.
          sampled: is a Tensor of shape [T, N, C] where C corresponds to
               `discretization_tot_classes` as given to the constructor.
                This will be needed for the 'reverse' model.
          softmax: is a "soft" version of `sampled`; if None, will default to `sampled`.
          token_ids:
            A list-of-list IDs. Each sublist contains IDs for an utterance.
            The IDs can be either phone IDs or word piece IDs.
          sos_id:
            sos token id
          eos_id:
            eos token id
          padding_id:
            token id used for padding of the `token_ids` when they appear as the
            input, e.g. blank id or eos_id.
        Returns:
            A scalar, the **sum** of label smoothing loss over utterances
            in the batch without any normalization.
        """

        # Add both sos and eos symbols to token_ids.  These will be used
        # as an input, there is no harm in adding both of these.
        token_ids_tensors = [ torch.tensor([sos_id] + utt + [eos_id]) for utt in token_ids ]

        tokens_padded = pad_sequence(token_ids_tensors, batch_first=True,
                                     padding_value=padding_id).to(positive_embed_shifted.device)

        tokens_key_padding_mask = decoder_padding_mask(tokens_padded, ignore_id=padding_id)

        # Let S be the length of the longest sentence (padded)
        token_embedding = self.token_embed(tokens_padded) * self.token_embed_scale # (N, S) -> (N, S, C)
        # add absolute position-encoding information
        token_embedding = self.abs_pos(token_embedding)

        token_embedding = token_embedding.permute(1, 0, 2)  # (N, S, C) -> (S, N, C)

        token_memory = self.reverse_encoder(token_embedding,
                                            src_key_padding_mask=tokens_key_padding_mask)
        # token_memory is of shape (S, N, C), if S is length of token sequence.

        T = positive_embed_shifted.shape[0]
        # the targets, here, are the hidden discrete symbols we are predicting
        tgt_mask = generate_square_subsequent_mask(T, device=positive_embed_shifted.device)

        hidden_predictor = self.reverse_decoder(
            tgt=positive_embed_shifted,
            memory=token_memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=tokens_key_padding_mask)

        total_prob = self.sample_and_predict.compute_prob(
            hidden_predictor,
            sampled,
            softmax,
            memory_key_padding_mask)

        # TODO: consider using a label-smoothed loss.
        return total_prob


class FastOffsetLayer(nn.Module):
    """
    A layer that rapidly learns an offset/bias on its output
    """
    def __init__(self,
                 dim: int,
                 bias_scale: float = 100.0):
        super(FastOffsetLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.bias_scale = bias_scale

    def forward(self, x):
        """
        An offset is added, treating the last dim of x as the channel dim.
        """
        if random.random() < 0.005:
            print("bias = ", self.bias)
        return x + self.bias * self.bias_scale




class SimpleCausalEncoderLayer(nn.Module):
    """
    This is a simple encoder layer that only sees left-context; it is
    based on the ConformerEncoderLayer, with the attention and one of
    the feed-forward components stripped out.
    """
    def __init__(self,
                 d_model: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 cnn_module_kernel: int = 15):
        super(SimpleCausalEncoderLayer, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.conv_module = CausalConvolutionModule(d_model,
                                                   cnn_module_kernel)

        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.ff_scale = 0.5
        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(d_model)  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tensor:
        # convolution module
        residual = src
        src = self.norm_conv(src)
        src = residual + self.dropout(self.conv_module(src))

        # feed forward module
        residual = src
        src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))

        # final normalization
        src = self.norm_final(src)
        return src


class ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, x_grad):
        return -x_grad

def reverse_gradient(x: Tensor) -> Tensor:
    return ReverseGrad.apply(x)


class DebugGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.save_for_backward(x)
        ctx.name = name
        return x
    @staticmethod
    def backward(ctx, x_grad):
        x, = ctx.saved_tensors
        x_grad_sum = x_grad.sum().to('cpu').item()
        x_grad_x_sum = (x_grad * x).sum().to('cpu').item()
        print(f"For {ctx.name}, x_grad_sum = {x_grad_sum}, x_grad_x_sum = {x_grad_x_sum}")
        return x_grad, None



class SampleAndPredict(nn.Module):
    """This module discretizes its input and lets you predict the
    discrete classes.
    We use the torch-flow-sampling package for this, to provide a differentiable
    softmax that should be much better than Gumbel in terms of actually giving
    an information bottleneck.
    (However, if straight_through_scale == 1.0, which actually seems
    to be working fine so far, it's the same as just sampling from the
    categorical distribution and using straight-through derivatives
    (i.e. the derivatives are as if that output had just been a softmax).
    This may depend somewhat on the model; straight_through_scale == 0.0
    is definitely safer from a correctness point of view.

    Args:
        dim:  The input feature dimension
        tot_classes:  The total number of classes (across all groups
             of classes); each group is separately discretized
        num_groups:  The number of groups of classes; discretization
             is done separately within each group.
        interp_prob: The probability with which we interpolate
             between two classes, assuming the sampling picks
             to distinct classes.  Making this smaller would give
             more noisy derivatives but makes the operation closer
             to a true sampling operation.  However, even with
             interp_prob = 1.0, as the distribution gets very
             peaky we'll still mostly have a single class in
             the output.
        straight_through_scale: The scale on the "straight-through"
             derivatives, in which we treat the softmax derivatives
             as the derivatives of the softmax+sampling+interpolation
             operation.  This, and interp_prob, may need to be
             changed as you train, just directly set the
             variable self._straight_through_scale if you need to.
             The "true" derivative will be scaled as
             1.0 - straight_through_scale.
        min_prob_ratio: For any class whose average softmax
             output, for a given minibatch, is less than
             min_prob_ratio times the average probability,
             boost its probability; this is a mechanism
             to avoid "losing" classes, we are hoping it won't really
             be necessary in practice.
    """
    def __init__(
            self,
            dim: int,
            tot_classes: int,
            num_groups: int,
            interp_prob: float = 1.0,
            straight_through_scale: float = 0.0,
            min_prob_ratio: float = 0.1,
            ):
        super(SampleAndPredict, self).__init__()

        self.linear1 = nn.Linear(dim, tot_classes)

        self.num_groups = num_groups
        self.interp_prob = interp_prob
        self.straight_through_scale = straight_through_scale
        self.min_prob_ratio = min_prob_ratio
        self.tot_classes = tot_classes
        self.classes_per_group = tot_classes // num_groups

        # prob_boost relates to the min_prob_ratio setting.  It's not configurable for now.
        self.prob_boost = 1.0e-03

        # class_probs is a rolling mean of the output of the sampling operation.
        # When any element of it gets below self.min_prob_ratio / self.classes_per_group,
        # we boost the class's probability by adding self.prob_boost to
        # that element of self.class_offset
        self.class_probs_decay = 0.95
        self.register_buffer('class_probs', torch.ones(tot_classes) / self.classes_per_group)
        # class_offsets is a bias term that we add to logits before the sampling
        # operation in order to enforce that no class is too infrequent
        # (c.f. 'min_prob_ratio').
        self.register_buffer('class_offsets', torch.zeros(tot_classes))


        # pred_linear predicts the class probabilities from a predictor
        # embedding of dimension 'dim' supplied by the user.
        self.pred_linear = nn.Linear(dim, tot_classes)

        if self.num_groups > 1:
            # We predict the logprobs of each group from the outputs of the
            # previous groups.  This is done via a masked multiply, where
            # the masking operates on blocks.  This projects from [all but
            # the last group] to [all but the first group], so the diagonal
            # of the mask can be 1, not 0, saving compute..
            d = tot_classes - self.classes_per_group
            c = self.classes_per_group
            self.pred_cross = nn.Parameter(torch.zeros(d, d))
            # If d == 4 and c == 2, the expression below has the following value
            # (treat True as 1 and False as 0).
            #tensor([[ True,  True, False, False],
            #        [ True,  True, False, False],
            #        [ True,  True,  True,  True],
            #        [ True,  True,  True,  True]])
            self.register_buffer('pred_cross_mask',
                                 ((torch.arange(d) // c).unsqueeze(1) >=
                                  (torch.arange(d) // c).unsqueeze(0)))

        # linear2 and post_layer_norm come after the sampling.
        self.linear2 = nn.Linear(tot_classes, dim)
        self.post_layer_norm = nn.LayerNorm(dim)

        self._reset_parameters()

    def _reset_parameters(self):
        if hasattr(self, 'pred_cross'):
            torch.nn.init.kaiming_uniform_(self.pred_cross, a=math.sqrt(5))


    def forward(self, x: Tensor, need_softmax: bool = True) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Forward computation.  See also compute_prob().
        Args:
              x:  The input tensor, of shape (S, N, E) where S is the sequence length,
                  N is the batch size and E is the embedding dim.
        Returns (sampled, softmax, positive_embed, negative_embed), where:

              sampled: of shape (S, N, C) where C is the `tot_classes` to the
                constructor, these are the sampled one-hot vectors or interpolations
                thereof.  They will be needed if we try to predict the discrete values
                (e.g. some kind of reverse model).
              softmax: A Tensor of shape (S, N, C) if need_softmax is True; else, None.
                This is the  non-sampled softmax output.  We use this as the target when
                evaluating the 'reverse' model (predicting the probabilities of these
                classes), as we can treat it as an
                expectation of the result of sampling -> lower-variance derivatives.
                This is unnecessary if straight_through_scale == 1.0, since in that
                case it would not affect the backpropagated derivatives.
              positive_embed: The samples projected back down to the embedding
                 dimension, and layer-normed (`dim` passed to the constructor).
              negative_embed: This is numerically the same value as positive_embed,
                 but has its gradients reversed prior to the projection and
                 LayerNorm.  This is intended to be used for terms that appear
                 with opposite sign in the loss function, to be fed to
                 something whose gradient is already (going to be) reversed:
                 specifically, the self-prediction network.
        """
        x = self.linear1(x) * 3

        if self.min_prob_ratio > 0.0:
            x = x + self.class_offsets

        (S, N, tot_classes) = x.shape

        x = x.reshape(S, N, self.num_groups, self.classes_per_group)

        # This is a little wasteful since we already compute the softmax inside
        # 'flow_sample'.
        softmax = x.softmax(dim=3).reshape(S, N, tot_classes) if need_softmax else None

        if random.random() < 0.05:
            # Some info that's useful for debug.
            softmax_temp = softmax.reshape(S, N, self.num_groups, self.classes_per_group)
            logsoftmax_temp = (softmax_temp + 1.0e-20).log()
            negentropy = (softmax_temp * logsoftmax_temp).sum(-1).mean()

            global_softmax = softmax_temp.mean(dim=(0,1))
            global_log_softmax = (global_softmax + 1.0e-20).log()
            global_negentropy = (global_softmax * global_log_softmax).sum(-1).mean()

            print("SampleAndPredict: entropy = ",
                  -negentropy.to('cpu').item(), ", averaged entropy = ",
                  -global_negentropy.to('cpu').item(),
                  ", argmax = ", (global_softmax * global_log_softmax).argmax(dim=-1).to('cpu'))


        x = torch_flow_sampling.flow_sample(x,
                                            interp_prob=self.interp_prob,
                                            straight_through_scale=self.straight_through_scale)

        assert x.shape == (S, N, self.num_groups, self.classes_per_group)
        x = x.reshape(S, N, tot_classes)

        sampled = x

        if self.training and self.min_prob_ratio > 0.0:
            mean_class_probs = torch.mean(x.detach(), dim=(0,1))
            self.class_probs = (self.class_probs * self.class_probs_decay +
                                mean_class_probs * (1.0 - self.class_probs_decay))
            prob_floor = self.min_prob_ratio / self.classes_per_group
            self.class_offsets += (self.class_probs < prob_floor) * self.prob_boost


        positive_embed = self.post_layer_norm(self.linear2(sampled))
        negative_embed = self.post_layer_norm(self.linear2(reverse_gradient(sampled)))

        if random.random() < 0.002:
            return (DebugGrad.apply(sampled, "sampled"), DebugGrad.apply(softmax, "softmax"),
                    positive_embed, negative_embed)
        else:
            return (sampled, softmax, positive_embed, negative_embed)


    def compute_prob(self, x: Tensor, sampled: Tensor, softmax: Optional[Tensor],
                     padding_mask: Optional[Tensor] = None,
                     reverse_grad: bool = False) -> Tensor:
        """
        Compute the total probability of the sampled probabilities, given
        some kind of predictor x (which we assume should not have access
        to the output on the current frame, but might have access to
        those of previous frames).

          x:  The predictor tensor, of shape (S, N, E) where S is the
              sequence length, N is the batch size and E is the embedding dim
              (`dim` arg to __init__()).  This is projected from `sampled`
              with a learnable matrix.
          sampled: A tensor of shape (S, N, C) where C is the `tot_classes`
              to the constructor, containing the sampled probabilities.
          softmax: A tensor of shape (S, N, C), this is the "smooth" version
              of `sampled`, which we use as the target in order to get
              lower-variance derivatives with the same expectation.
              If not provided, will default to `sampled`.
          padding_mask: Optionally, a boolean tensor of shape (N, S), i.e.
              (batch_size, sequence_length), with True in masked positions
              that are to be ignored in the sum of probabilities.
          reverse_grad:  If true, negate the gradient that is passed back
              to 'x' and to the modules self.pred_linear and pred_cross.
              This will be useful in computing a loss function that has
              a likelihood term with negative sign (i.e. the self-prediction).
              We'll later need to negate this gradient one more more time
              (it's expected, when reverse_grad == True, that x would
              derive somehow from `negative_embed`, so that the gradient
              will eventually go back to the correct sign.)

        Returns a scalar Tensor represnting the total probability.
        """
        if reverse_grad:
            sampled = reverse_gradient(sampled)
        if softmax is None:
            softmax = sampled
        elif reverse_grad:
            softmax = reverse_gradient(softmax)

        logprobs = self.pred_linear(x)

        # Add "cross-terms" to logprobs; this is a regression that uses earlier
        # groups to predict later groups
        if self.num_groups > 1:
            pred_cross = self.pred_cross * self.pred_cross_mask
            t = self.tot_classes
            c = self.classes_per_group
            # all but the last group.  Note: we could possibly use softmax here,
            # to reduce variance, but I was concerned about information leakage.
            sampled_in_part = sampled[:,:,0:t-c]
            # row index of pred_cross corresponds to output, col to input -> must transpose
            # before multiply.
            cross_out = torch.matmul(sampled_in_part, pred_cross.transpose(0, 1))
            # add the output of this matrix multiplication to all but the first
            # group in `logprobs`.  Each group is predicted based on previous
            # groups.
            logprobs[:,:,c:] += cross_out
        (S, N, C) = logprobs.shape
        logprobs = logprobs.reshape(S, N, self.num_groups, self.classes_per_group)
        # Normalize the log-probs (so they sum to one)
        logprobs = torch.nn.functional.log_softmax(logprobs, dim=-1)
        logprobs = logprobs.reshape(S, N, C)

        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool and padding_mask.shape == (N, S)
            padding_mask = torch.logical_not(padding_mask).transpose(0, 1).unsqueeze(-1)
            assert padding_mask.shape == (S, N, 1)
            tot_prob = (logprobs * softmax * padding_mask).sum()
        else:
            tot_prob = (logprobs * softmax).sum()

        if reverse_grad:
            tot_prob = reverse_gradient(tot_prob)
        return tot_prob


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = RelPositionMultiheadAttention(
            d_model, nhead, dropout=0.0,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)

        self.norm_ff_macaron = nn.LayerNorm(
            d_model
        )  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(
            d_model
        )  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            attn_mask: the mask for the src sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            attn_mask: (S, S).  This probably won't be used, in fact should not
                 be (e.g. could in principle ensure causal behavior, but
                  actually the conformer does not support this).
            key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        # macaron style feed forward module
        residual = src
        src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(
            self.feed_forward_macaron(src)
        )

        # multi-headed self-attention module
        residual = src
        src = self.norm_mha(src)
        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = residual + self.dropout(src_att)

        # convolution module
        residual = src
        src = self.norm_conv(src)
        src = residual + self.dropout(self.conv_module(src))

        # feed forward module
        residual = src
        src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))

        src = self.norm_final(src)

        return src


class ConformerEncoder(nn.Module):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super(ConformerEncoder, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args
            x: input of shape (T, N, C), i.e. (seq_len, batch, channels)
            pos_emb: positional embedding tensor of shape (1, 2*T-1, C),
            attn_mask (optional, likely not used): mask for self-attention of
                  x to itself, of shape (T, T)
            key_padding_mask (optional): mask of shape (N, T), dtype must be bool.
        Returns:
            Returns a tensor with the same shape as x, i.e. (T, N, C).
        """
        for mod in self.layers:
            x = mod(
                x,
                pos_emb,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        return x


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
            self.embed_dim,
            self.num_heads,
            self.in_proj.weight,
            self.in_proj.bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )

    def rel_shift(self, x: Tensor) -> Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape
        assert n == 2 * time1 - 1
        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, time1, time1),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(
                query, in_proj_weight, in_proj_bias
            ).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct."
                    )
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError(
                        "The size of the 3D attn_mask is not correct."
                    )
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()
                    )
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + self.pos_bias_v).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(
            q_with_bias_u, k
        )  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p.transpose(-2, -1)
        )  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd)

        attn_output_weights = (
            matrix_ac + matrix_bd
        ) * scaling  # (batch, head, time1, time2)

        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, -1
        )

        assert list(attn_output_weights.size()) == [
            bsz * num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)

        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(
            attn_output, out_proj_weight, out_proj_bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

class CausalConvolutionModule(nn.Module):
    """Modified from ConvolutionModule from the in Conformer model.
    This is a causal version of it (sees only left-context).

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).
    """

    def __init__(
        self, channels: int, kernel_size: int, bias: bool = True
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(CausalConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.kernel_size = kernel_size

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,  # We'll pad manually
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x: Tensor) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        (B, C, T) = x.shape
        padding = self.kernel_size - 1
        x = torch.cat((torch.zeros(B, C, padding, dtype=x.dtype, device=x.device), x),
                      dim=2)
        x = self.depthwise_conv(x)  # <-- This convolution module does no padding,
                                    # so we padded manually, on the left only.

        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)  # (batch, channels, time)

        return x.permute(2, 0, 1)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self, channels: int, kernel_size: int, bias: bool = True
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x: Tensor) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1) # (time, batch channel)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def identity(x):
    return x



def _gen_rand_tokens(N: int) -> List[List[int]]:
    ans = []
    for _ in range(N):
        S = random.randint(1, 20)
        ans.append([random.randint(3, 30) for _ in range(S)])
    return ans

def _gen_supervision(tokens: List[List[int]]):
    ans = dict()
    N = len(tokens)
    ans['sequence_idx'] = torch.arange(N, dtype=torch.int32)
    ans['start_frame'] = torch.zeros(N, dtype=torch.int32)
    ans['num_frames'] = torch.tensor([ random.randint(20, 35) for _ in tokens])
    return ans

def _test_bidirectional_conformer():
    num_features = 40
    num_classes = 1000
    m = BidirectionalConformer(num_features, num_classes)
    T = 35
    N = 10
    C = num_features
    feats = torch.randn(N, T, C)
    feats.requires_grad = True

    tokens = _gen_rand_tokens(N)
    supervision = _gen_supervision(tokens)
    print("tokens = ", tokens)
    print("supervision = ", supervision)
    # memory: [T, N, C]
    (memory, pos_emb, key_padding_mask) = m(feats, supervision)

    # ctc_output: [N, T, C].
    ctc_output = m.ctc_encoder_forward(memory, pos_emb, key_padding_mask)

    (sampled, softmax, positive_embed_shifted, negative_embed_shifted) = m.sample_forward(memory)

    decoder_logprob = m.decoder_forward(memory, key_padding_mask, tokens,
                                     sos_id=1,
                                     eos_id=2)
    print("decoder logprob = ", decoder_logprob)

    reverse_decoder_logprob = m.reverse_decoder_forward(
        positive_embed_shifted, key_padding_mask,
        sampled, softmax, tokens,
        sos_id=1, eos_id=2, padding_id=0)

    print("reverse decoder logprob = ", reverse_decoder_logprob)

    self_prediction_logprob = m.self_prediction_forward(
        negative_embed_shifted, key_padding_mask,
        sampled, softmax)

    print("self prediction logprob = ", self_prediction_logprob)

    loss = -(decoder_logprob + reverse_decoder_logprob - self_prediction_logprob)
    loss.backward()


def _test_discrete_bottleneck():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dim = 256
    tot_classes = 256
    num_groups = 8
    interp_prob = 1.0
    straight_through_scale = 1.0  # will change
    need_softmax = True

    b = SampleAndPredict(dim, tot_classes, num_groups,
                         interp_prob, straight_through_scale).to(device)

    from_feats_predictor = nn.Linear(dim, dim).to(device)

    from_negative_embed_predictor = nn.Linear(dim, dim).to(device)

    model = nn.ModuleList([b, from_feats_predictor, from_negative_embed_predictor])
    model.train()

    optim = torch.optim.Adam(params=model.parameters(),
                             lr=3.0e-04)


    scale = 0.5  # determines the feature correlation..should be between 0 and 1.
    #https://en.wikipedia.org/wiki/Mutual_information#Linear_correlation, -0.5 log(1 - rho^2)..
    # scale corresponds to rho^2, rho being sqrt(scale).
    mutual_information = dim * -0.5 * math.log(1.0 - scale)
    print("mutual_information = ", mutual_information)

    for epoch in range(10):
        torch.save(model.state_dict(), f'epoch-{epoch}.pt')
        for i in range(2000):
            # TODO: also test padding_mask
            T = 300
            N = 10

            feats = torch.randn(T, N, dim, device=device)

            feats2 = (feats * scale ** 0.5) + ((1.0 - scale) ** 0.5 * torch.randn(T, N, dim, device=device))

            #print(f"norm(feats) ={feats.norm()} vs. norm(feats2) = {feats2.norm()}")

            sampled, softmax, positive_embed, negative_embed = b(feats)

            E = dim
            negative_embed_shifted = torch.cat((torch.zeros(1, N, E).to(device),
                                                negative_embed[:-1,:,:]), dim=0)
            positive_embed_shifted = torch.cat((torch.zeros(1, N, E).to(device),
                                                positive_embed[:-1,:,:]), dim=0)

            # using feats2 instead of feats will limit the mutual information,
            # to the MI between feats and feats2, which we computed and printed
            # above as mutual_information.
            predictor = from_feats_predictor(feats2)

            prob = b.compute_prob(predictor, sampled, softmax)

            if True:
                predictor_shifted = from_negative_embed_predictor(negative_embed_shifted)

                self_prob = b.compute_prob(predictor_shifted, sampled, softmax,
                                           reverse_grad=True)
                normalized_self_prob = (self_prob / (T * N))

            normalized_prob = (prob / (T * N))

            normalized_loss = -(normalized_prob - normalized_self_prob)

            if i % 200 == 0:
                print(f"Epoch {epoch}, iteration {i}, normalized loss/frame is {-normalized_prob.to('cpu').item()} - {-normalized_self_prob.to('cpu').item()} = {normalized_loss.to('cpu').item()}")

            normalized_loss.backward()

            optim.step()
            optim.zero_grad()


if __name__ == '__main__':
    _test_bidirectional_conformer()
    _test_discrete_bottleneck()
