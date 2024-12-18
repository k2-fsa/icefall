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


import random
import warnings
from typing import Dict, Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear, penalize_abs_values_gt
from torch import Tensor

from icefall.utils import add_sos, make_pad_mask


class PromptedTransducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        text_encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        use_BERT: bool = True,
        text_encoder_type: str = "BERT",
        text_encoder_adapter: bool = False,
        freeze_text_encoder: bool = True,
        context_fuser: nn.Module = None,
    ):
        """
        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          text_encoder:
            This is a encoder that processes text information (e.g content prompt
            and style prompt). The input is `x` of (N,T) and `x_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
          text_encoder_type:
            The type of the text_encoder. Supported are (BERT, DistilBERT)
          context_fuser
            A optional module that fuses the embeddings of text encoder. The fused embedding
            will be added to the joiner.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = ScaledLinear(
            encoder_dim,
            vocab_size,
            initial_scale=0.25,
        )
        self.simple_lm_proj = ScaledLinear(
            decoder_dim,
            vocab_size,
            initial_scale=0.25,
        )

        self.use_BERT = use_BERT  # if the text encoder is a pre-trained BERT
        self.context_fuser = context_fuser

        assert text_encoder_type in (
            "BERT",
            "DistilBERT",
            "BERT-UNCASED",
        ), f"Unseen text_encoder type {text_encoder_type}"
        self.text_encoder_dim = (
            self.text_encoder.config.hidden_size
            if text_encoder_type in ("BERT", "BERT-UNCASED")
            else self.text_encoder.config.dim
        )
        self.freeze_text_encoder = freeze_text_encoder

        if text_encoder_adapter:
            self.text_encoder_adapter = nn.Sequential(
                nn.Linear(self.text_encoder_dim, self.text_encoder_dim, bias=False),
                nn.Tanh(),
            )
        else:
            self.text_encoder_adapter = None

        self.style_prompt_embedding = nn.Parameter(
            torch.full((self.text_encoder_dim,), 0.5)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        encoded_inputs: Dict,
        style_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_pre_text: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          text:
            A 2-D tensor of integer dtype containing prompt text, of shape (N, T).
            It is exptected to contain the style prompt (first) and then the content
            prompt.
          text_lens:
            A 1-D tensor of shape (N,). It contains the number of elements (bytes)
            in `text` before padding, which will include the lengths of the
            style plus the content prompt.
          style_lens:
            A 1-D tensor of shape (N,), containing the number of elements (bytes)
            within each row of `text` that correspond to the style prompt (these
            are expected to come first).
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
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        x, x_lens = self.encoder_embed(x, x_lens)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # freeze the BERT text encoder

        if use_pre_text:
            memory, memory_key_padding_mask = self.encode_text(
                encoded_inputs, style_lens=style_lens
            )
        else:
            memory = None
            memory_key_padding_mask = None

        encoder_out, x_lens = self.encoder(
            x,
            x_lens,
            src_key_padding_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
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
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
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
        if self.context_fuser is not None and memory is not None:
            memory = memory.permute(1, 0, 2)  # (T,N,C) -> (N,T,C)
            context = self.context_fuser(memory, padding_mask=memory_key_padding_mask)
            context = self.joiner.context_proj(context)
        else:
            context = None

        logits = self.joiner(am_pruned, lm_pruned, context=context, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss)

    def _add_style_indicator(self, memory: Tensor, style_lens: Tensor):
        """
        Adds to `memory` an indicator that is 1.0 for positions that correspond to
        the `style prompt` and 0 elsewhere.  The scale can be fixed because the
        scale of the embedding vector can adjust to compensate.

        Args:
            memory: (memory_len, batch_size, embed_dim)
            style_lens: (batch_size,),  a vector of lengths of the style prompt.
        """

        (memory_len, batch_size, embed_dim) = memory.shape

        indicator = (
            torch.arange(memory_len, device=memory.device).unsqueeze(-1) < style_lens
        )
        indicator = indicator.to(memory.dtype)

        extra_term = torch.zeros_like(memory)
        extra_term += indicator.unsqueeze(-1) * self.style_prompt_embedding.expand(
            memory_len, batch_size, self.text_encoder_dim
        )

        return memory + extra_term

    def encode_text(
        self,
        encoded_inputs: Dict,
        style_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get the embeddings of text

        Args:
            encoded_inputs: The encoded inputs generated by a tokenizer (Dict)

        Returns:
            Tuple[Tensor, Tensor]: Returns the text embeddings encoded by the
            text_encoder and the attention mask
        """
        text_lens = encoded_inputs.pop("length")  # need to use pop to remove this item

        # Freeze the pre-trained text encoder
        with torch.no_grad():
            memory = self.text_encoder(**encoded_inputs)["last_hidden_state"]  # (B,T,C)
            memory = memory.permute(1, 0, 2)

        # Text encoder adapter
        if self.text_encoder_adapter is not None:
            memory = self.text_encoder_adapter(memory)

        memory = self._add_style_indicator(memory, style_lens)

        memory_key_padding_mask = make_pad_mask(text_lens)

        return memory, memory_key_padding_mask

    def encode_audio(
        self,
        feature: Tensor,
        feature_lens: Tensor,
        memory: Optional[Tensor],
        memory_key_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Encode the input audio features

        Args:
            feature (Tensor): Input audio (N,T,C)
            feature_lens (Tensor): Length of input audio (N,)
            memory (Tensor): Embeddings from the text encoder
            memory_key_padding_mask (Tensor): _description_

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        x, x_lens = self.encoder_embed(feature, feature_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(
            x=x,
            x_lens=x_lens,
            src_key_padding_mask=src_key_padding_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return encoder_out, encoder_out_lens


Transducer = PromptedTransducer  # for decoding
