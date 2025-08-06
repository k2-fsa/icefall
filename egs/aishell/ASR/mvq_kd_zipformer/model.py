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

from typing import Optional, Tuple, List

import sys
import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment
from scaling import ScaledLinear

from icefall.utils import add_sos, make_pad_mask, time_warp


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        attention_decoder: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        # use_transducer: bool = False,
        use_ctc: bool = False,
        use_attention_decoder: bool = False,
        num_codebooks: Tuple[int] = 0,
        middle_output_layers: Tuple[int] = None,  # 0-based layer index
        dk_layers_opt:str = "uncertainty",
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
          use_attention_decoder:
            Whether use attention-decoder head. Default: False.
          num_codebooks:
            Used by distillation loss.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        # multilayer distillation weight options
        if dk_layers_opt == "avg":
          self.average_loss = True
          self.uncertainty_loss = False
          print("dk layers opt is avg")

        if dk_layers_opt.startswith("uncertainty"):
          self.average_loss = False
          self.uncertainty_loss = True
          print("dk layers opt is uncertainty")

        self.uncertainty_opt = dk_layers_opt
        # multilayer distillation weight options

        self.middle_output_layers = middle_output_layers
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
            print(f"encoder dim is {encoder_dim}")
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

        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.attention_decoder = attention_decoder
        else:
            assert attention_decoder is None
        # Modules for mvq-kd  ------------------------------------------------------------------------------------------------------------------------
        if middle_output_layers is not None:  
          from icefall import is_module_available
          if not is_module_available("multi_quantization"):
              raise ValueError("Please 'pip install multi_quantization' first.")
          from multi_quantization.prediction import JointCodebookLoss, AutomaticWeightedLoss
          codebook_loss_nets = []
          self.num_codebooks = num_codebooks
          # single layer distillation
          if len(self.middle_output_layers) == 1 and self.middle_output_layers is not None:
            print ("start single layer distillation ... ")
            self.is_multilayer_distill = False
            if self.num_codebooks > 0:
                codebook_loss_net = JointCodebookLoss(
                    predictor_channels = (192 if self.middle_output_layers == 0 else self.encoder.encoder_dim[self.middle_output_layers - 1] if 1 <= self.middle_output_layers <= 6 else 256),
                    num_codebooks=self.num_codebooks,
                    is_joint=False,
                )
                codebook_loss_nets.append(codebook_loss_net)
            self.codebook_loss_nets = nn.ModuleList(codebook_loss_nets)
          # multiple layers distillation
          else:
            print ("start multiple layer distillation ... ")
            assert len(self.middle_output_layers) > 1 and self.middle_output_layers is not None, self.middle_output_layers
            if self.uncertainty_loss:
              self.awl = AutomaticWeightedLoss(2)  # codebook_loss1, codebook_loss2, simple_loss, pruned_rnnt_loss, 
            # if self.momentum:
            #   self.mwl = MomentumWeightedLoss()
            self.is_multilayer_distill = True
            for middle_output_layer_n, num_codebook in zip(self.middle_output_layers, self.num_codebooks):
                codebook_loss_net = JointCodebookLoss(
                    predictor_channels = (self.encoder.encoder_dim[0] if middle_output_layer_n == 0 else self.encoder.encoder_dim[middle_output_layer_n - 1] if 1 <= middle_output_layer_n <= 6 else self.encoder.encoder_dim[-1]),
                    num_codebooks=num_codebook,
                    is_joint=False,
                    enable_dynamic_temp=False,
                    # checkpoint=False,
                )
                # print("predictor_channels is : ",192 if middle_output_layer_n == 0 else self.encoder.encoder_dim[middle_output_layer_n - 1] if 1 <= middle_output_layer_n <= 6 else 256)
                codebook_loss_nets.append(codebook_loss_net)

            self.codebook_loss_nets = nn.ModuleList(codebook_loss_nets)
        # Modules for mvq-kd  ------------------------------------------------------------------------------------------------------------------------

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
        middle_layer_outputs = []  # 0:encoder_embedding,  1:Zipformer2Encoder,  2-6:DownsampledZipformer2Encoder
        middle_layer_outputs.append(x)  # 0     
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens, outputs = self.encoder(x, x_lens, src_key_padding_mask)
        # middle_layer_outputs.extend(outputs)
        for tensor in outputs:
          middle_layer_outputs.append(tensor.permute(1, 0, 2))

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        # for idx, tensor in enumerate(middle_layer_outputs):
        #     print(f"Layer {idx}: {tensor.shape if isinstance(tensor, torch.Tensor) else 'Non-tensor'}")
        # print("encoder_out shape is : ", encoder_out.shape)
        return encoder_out, encoder_out_lens, middle_layer_outputs

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

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
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
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_cr_ctc: bool = False,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        codebook_indexes: List[torch.Tensor] = None,
        teacher_weights_layers: List[List[float]] = [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[float], torch.Tensor]:
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
          codebook_indexes:
            codebook_indexes extracted from a teacher model.
          distillation_layer:
            number of distillation layers 0:encoder_embedding,  1:Zipformer2Encoder,  2-6:DownsampledZipformer2Encoder

        Returns:
          Return the transducer losses, CTC loss, AED loss,
          and consistency-regularization loss in form of
          (simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        device = x.device

        if use_cr_ctc:
            assert self.use_ctc
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

        # Compute encoder outputs
        encoder_out, encoder_out_lens, middle_layer_outputs = self.forward_encoder(x, x_lens)
        middle_layer_outputs.append(encoder_out)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
            if use_cr_ctc:
                simple_loss = simple_loss * 0.5
                pruned_loss = pruned_loss * 0.5
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            if not use_cr_ctc:
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
            ctc_loss = torch.empty(0)
            cr_loss = torch.empty(0)

        if self.use_attention_decoder:
            attention_decoder_loss = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y.to(device),
                ys_lens=y_lens.to(device),
            )
            if use_cr_ctc:
                attention_decoder_loss = attention_decoder_loss * 0.5
        else:
            attention_decoder_loss = torch.empty(0)

        # distillation part new ------------------------------------------------------------------------------------------------------------------------

        formatted_total_losses = []
        codebook_loss_sum = 0
        total_losses = []
        # if codebook_indexes is none, disable compute codebook_loss_sum
        if codebook_indexes is not None:
          if len(codebook_indexes) == 6:
            group1_indices = [0, 2, 4]  # 第1、3、5个张量，多教师第一层
            group2_indices = [1, 3, 5]  # 第2、4、6个张量，多教师第二层
            group1 = torch.stack([codebook_indexes[i] for i in group1_indices], dim=0)  # 形状 [3, 46, 322, 16]
            group2 = torch.stack([codebook_indexes[i] for i in group2_indices], dim=0)  # 形状 [3, 46, 161, 16]
            codebook_indexes = []
            codebook_indexes.append(group1)
            codebook_indexes.append(group2)
          if len(codebook_indexes) == 2:
            for middle_output_layer, codebook_loss_net, codebook_index, teacher_weights in zip(self.middle_output_layers, self.codebook_loss_nets, codebook_indexes, teacher_weights_layers):
              if middle_output_layer  is not  None  and  middle_output_layer < 7:
                assert isinstance(middle_output_layer, int), f"middle_output_layer type :  {type(middle_output_layer).__name__}"
                middle_layer_output = middle_layer_outputs[middle_output_layer]
              
              if middle_output_layer  is not  None  and  middle_output_layer == 7:
                assert isinstance(middle_output_layer, int), f"middle_output_layer type :  {type(middle_output_layer).__name__}"
                middle_layer_output = encoder_out

              if self.training and codebook_index is not None:
                  assert hasattr(self, "codebook_loss_nets")
                  # print(f"Layer {middle_output_layer} ----->> codebook_index shape : {codebook_index.shape}, middle_layer_output shape : {middle_layer_output.shape}")
                  if codebook_index.shape[-2] != middle_layer_output.shape[-2]:
                    print("batch_size of student not match teacher")
                    sys.exit(1)
                  codebook_loss = codebook_loss_net(
                      middle_layer_output, codebook_index, teacher_weights
                  )
                  assert self.average_loss != self.uncertainty_loss,f"self.average_loss is {self.average_loss}  self.uncertainty_loss is {self.uncertainty_loss}"
                  if self.average_loss:
                    codebook_loss_sum += codebook_loss / len(self.middle_output_layers)
                  if self.uncertainty_loss:
                    total_losses.append(codebook_loss)
                    if len(total_losses) == len(self.middle_output_layers):
                      total_losses, weights = self.awl(total_losses, self.uncertainty_opt) # uncertainty weight
                      assert len(total_losses) == len(weights) == 2, f"total_losses lengh is {len(total_losses)}         weights lengh is {len(weights)}"
                      sigma_values = [w.item() for w in weights]
                      print(f"uncertainty sigma is : {sigma_values}")
                      for loss, weight  in zip(total_losses, weights):
                        codebook_loss_sum += loss
              else:
                  # when codebook index is not available.
                  codebook_loss_sum = None
          # distillation part new -----------------------------------------------------------------------------------------------------------------------

        return simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss, middle_layer_outputs, formatted_total_losses, codebook_loss_sum





    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes):
        # Output rate of hubert is 50 frames per second,
        # while that of current encoder is 25.
        # Following code handling two issues:
        # 1.
        #   Roughly speaking, to generate another frame output,
        #   hubert needes extra two frames,
        #   while current encoder needs extra four frames.
        #   Suppose there are only extra three frames provided,
        #   hubert will generate another frame while current encoder does nothing.
        # 2.
        #   codebook loss is a frame-wise loss, to enalbe 25 frames studnet output
        #   learns from 50 frames teacher output, two successive frames of teacher model
        #   output is concatenated together.
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape

        # Handling issue 1.
        if T >= t_expected * 2:
            codebook_indexes = codebook_indexes[:, : t_expected * 2, :]
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * 2)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes