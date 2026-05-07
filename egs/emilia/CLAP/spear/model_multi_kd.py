# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# Copyright    2024 University of Cambridge      (authors: Xiaoyu Yang)
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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_quantization.prediction import JointCodebookLoss

from icefall.utils import make_pad_mask


class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_dim: int,
        num_codebooks: int=8,
        distillation_layer: int=9,
        distillation_delta: int=0,
        teacher_frame_ratio: int = 2,
        interpolate_teacher: bool = False,
        num_events: int = 527
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

        
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
            
        self.distillation_layer = distillation_layer
        # the frame ratio between the teacher and student
        # if larger than one, we are basically having more than one set of
        # codebooks for each frame
        self.num_codebooks= num_codebooks
        self.teacher_frame_ratio = teacher_frame_ratio 
        self.interpolate_teacher = interpolate_teacher
        self.distillation_delta = distillation_delta
        
        if num_codebooks > 0:
            self.codebook_loss_net = JointCodebookLoss(
                predictor_channels=encoder_dim,
                num_codebooks=num_codebooks * self.teacher_frame_ratio,
                is_joint=False,
                reduction="none",
            )
        else:
            self.codebook_loss_net = None
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes

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

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        codebook_indexes: torch.Tensor = None,
        at_targets: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          codebook_indexes:
            Codebook indexes of teacher embeddings
            
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
        assert codebook_indexes is not None or at_targets is not None

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
            
        if codebook_indexes is not None and self.codebook_loss_net is not None:
            codebook_loss = self.forward_codebook_loss(encoder_out, encoder_out_lens, codebook_indexes)
        else:
            codebook_loss = None
        
        if at_targets is not None:
            at_loss = self.forward_audio_tagging(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        else:
            at_loss = None
        
        return codebook_loss, at_loss

    def forward_codebook_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        codebook_indexes: torch.Tensor,
    ):
        # align the encoder features with the codebook indexes
        if self.interpolate_teacher:
            codebook_indexes = self.interpolate_codebook_indexes(
                encoder_out, codebook_indexes
            )
        else:
            if codebook_indexes.shape[1] != encoder_out.shape[1]:
                # align the codebook indexes to the frame rate of the student encoder out
                codebook_indexes = self.concat_successive_codebook_indexes(
                    encoder_out, codebook_indexes, ratio=self.teacher_frame_ratio
                )
                
        # the delta is associated with the frame-rate of the encoder
        # so a bigger delta maybe necessary for 50Hz student encoder
        if self.distillation_delta > 0:
            codebook_indexes = codebook_indexes[:,:-self.distillation_delta, :]
            encoder_out = encoder_out[:, self.distillation_delta:, :]
            truncated_padding_mask = make_pad_mask(encoder_out_lens - self.distillation_delta)
            codebook_indexes = codebook_indexes.masked_fill(truncated_padding_mask.unsqueeze(-1), value=-100)
            
        N,T,_ = encoder_out.shape
        codebook_loss = self.codebook_loss_net(encoder_out.float(), codebook_indexes)
        codebook_loss = codebook_loss.reshape(N,T,-1)
        num_cb = codebook_loss.size(-1)
        # normalize the loss by the number of codebooks
        codebook_loss = codebook_loss.sum(dim=(1,2)) / num_cb
        
        return codebook_loss

    def forward_audio_tagging(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        target: torch.Tensor = None,
        return_logits: bool = False,
    ):
        # target: (N, num_events)
        logits = self.audio_tagging_proj(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) # (N,T)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits) # (N, num_events)
        if return_logits:
            return logits
        
        at_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        return at_loss
    
    @staticmethod
    def interpolate_codebook_indexes(middle_layer_output, codebook_indexes):
        # This function addresses the case where the teacher has a lower frame rate
        # than the student model
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape # C should be 256
        
        codebook_indexes = codebook_indexes.permute(0,2,1).float() # (N,C,T)
        codebook_indexes = torch.nn.functional.interpolate(codebook_indexes, t_expected)
        codebook_indexes = codebook_indexes.permute(0,2,1).int() # (N,T,C)
        
        assert codebook_indexes.shape[1] == middle_layer_output.shape[1]
        return codebook_indexes
    
    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes, ratio=2):
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
        N, T, C = codebook_indexes.shape # C should be 256
        
        # Handling issue 1.
        if T >= t_expected * ratio:
            codebook_indexes = codebook_indexes[:, : t_expected * ratio, :]
        else:
            assert t_expected * ratio - T <= 5, (T, t_expected, ratio)
            diff = t_expected * ratio - T
            codebook_indexes = torch.cat(
                [
                    codebook_indexes,
                    torch.full((N,diff,C), -100).to(codebook_indexes.device).to(codebook_indexes.dtype)
                ],
                dim=1,
            )
        assert codebook_indexes.size(1) == middle_layer_output.size(1) * ratio
        
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
    
class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_dim: int = 384,
        num_events: int = 527,
    ):
        """An audio tagging model

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
          num_event:
            The number of classes.
        """
        super().__init__()

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        )

        # for multi-class classification
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

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
            The ground truth label of audio events, could be many hot
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        # Forward the speaker module
        logits = self.forward_audio_tagging(
            encoder_out=encoder_out, encoder_out_lens=encoder_out_lens
        )  # (N, num_classes)

        loss = self.criterion(logits, target)

        return loss

    def forward_audio_tagging(self, encoder_out, encoder_out_lens):
        """
        Args:
          encoder_out:
            A 3-D tensor of shape (N, T, C).
          encoder_out_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          A 3-D tensor of shape (N, num_classes).
        """
        logits = self.classifier(encoder_out)  # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)  # mask the padding frames
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(
            logits
        )  # normalize the logits

        return logits