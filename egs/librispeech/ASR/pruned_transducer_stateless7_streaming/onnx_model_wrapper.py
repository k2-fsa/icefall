import argparse
import logging
import os
from typing import Optional, Tuple

import sentencepiece as spm
import torch
from torch import nn

import torch.nn.functional as F
from scaling import ScaledConv1d, ScaledEmbedding, ScaledLinear


from model import Transducer

from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    setup_logger,
    str2bool,
)
from icefall.utils import is_jit_tracing, make_pad_mask

class OnnxStreamingEncoder(torch.nn.Module):
    """
    Args:
          left_context:
            How many previous frames the attention can see in current chunk.
            Note: It's not that each individual frame has `left_context` frames
            of left context, some have more.
          right_context:
            How many future frames the attention can see in current chunk.
            Note: It's not that each individual frame has `right_context` frames
            of right context, some have more.
          chunk_size:
            The chunk size for decoding, this will be used to simulate streaming
            decoding using masking.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        len_cache: torch.tensor,
        avg_cache: torch.tensor,
        attn_cache: torch.tensor,
        cnn_cache: torch.tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states:
            The decode states for previous frames which contains the cached data.
            It has two elements, the first element is the attn_cache which has
            a shape of (encoder_layers, left_context, batch, attention_dim),
            the second element is the conv_cache which has a shape of
            (encoder_layers, cnn_module_kernel-1, batch, conv_dim).
            Note: states will be modified in this function.
          processed_lens:
            How many frames (after subsampling) have been processed for each sequence.

        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
            - decode_states, the updated states including the information
              of current chunk.
        """
        num_encoder_layers = []
        encoder_attention_dims = []
        states = []
        for i, encoder in enumerate(self.model.encoders):
            num_encoder_layers.append(encoder.num_layers)
            encoder_attention_dims.append(encoder.attention_dim)

        len_cache = len_cache.transpose(0,1) # [sum(num_encoder_layers), B]
        offset = 0
        for num_layer in num_encoder_layers:
            states.append(len_cache[offset:offset+num_layer])
            offset += num_layer

        avg_cache = avg_cache.transpose(0,1) # [15, B, 384]
        offset = 0
        for num_layer in num_encoder_layers:
            states.append(avg_cache[offset:offset+num_layer])
            offset += num_layer

        attn_cache = attn_cache.transpose(0,2) # [15*3, 64, B, 192]
        left_context_len = attn_cache.shape[1]
        offset = 0
        for i, num_layer in enumerate(num_encoder_layers):
            ds = self.model.zipformer_downsampling_factors[i]
            states.append(attn_cache[offset:offset+num_layer, :left_context_len // ds])
            offset += num_layer
        for i, num_layer in enumerate(num_encoder_layers):
            encoder_attention_dim = encoder_attention_dims[i]
            ds = self.model.zipformer_downsampling_factors[i]
            states.append(attn_cache[offset:offset+num_layer, :left_context_len // ds, :, :encoder_attention_dim // 2])
            offset += num_layer
        for i, num_layer in enumerate(num_encoder_layers):
            ds = self.model.zipformer_downsampling_factors[i]
            states.append(attn_cache[offset:offset+num_layer, :left_context_len // ds, :, :encoder_attention_dim // 2])
            offset += num_layer

        cnn_cache = cnn_cache.transpose(0,1) # [30, B, 384, cnn_kernel-1]
        offset = 0
        for num_layer in num_encoder_layers:
            states.append(cnn_cache[offset:offset+num_layer])
            offset += num_layer
        for num_layer in num_encoder_layers:
            states.append(cnn_cache[offset:offset+num_layer])
            offset += num_layer

        encoder_out, encoder_out_lens, new_states = self.model.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=states,
        )

        new_len_cache = torch.cat(states[:self.model.num_encoders]).transpose(0,1) # B,15
        new_avg_cache = torch.cat(states[self.model.num_encoders:2*self.model.num_encoders]).transpose(0,1) # [B,15,384]
        new_cnn_cache = torch.cat(states[5*self.model.num_encoders:]).transpose(0,1) # [B,2*15,384,cnn_kernel-1]
        assert len(set(encoder_attention_dims)) == 1
        # pad_tensors = [tensor.expand(-1,left_context_len,-1,encoder_attention_dims[0]) for tensor in states[2*self.model.num_encoders:5*self.model.num_encoders]]
        pad_tensors = [torch.nn.functional.pad(tensor,(0,encoder_attention_dims[0]-tensor.shape[-1],0,0,0,left_context_len-tensor.shape[1],0,0)) for tensor in states[2*self.model.num_encoders:5*self.model.num_encoders]]
        new_attn_cache = torch.cat(pad_tensors).transpose(0,2) # [B,64,15*3,192]

        return encoder_out, encoder_out_lens, new_len_cache, new_avg_cache, new_attn_cache, new_cnn_cache

class OnnxDecoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = ScaledEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
            padding_idx=blank_id,
        )
        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = ScaledConv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim,
                bias=False,
            )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        embedding_out = self.embedding(y)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)

                # During inference time, there is no need to do extra padding
                # as we only need one output
            assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out

class OnnxJoiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        if not is_jit_tracing():
            assert encoder_out.ndim == decoder_out.ndim
            assert encoder_out.ndim in (2, 4)
            assert encoder_out.shape == decoder_out.shape

        
        logit = self.encoder_proj(encoder_out) + self.decoder_proj(
                decoder_out
            )
            
        logit = self.output_linear(torch.tanh(logit))

        return logit