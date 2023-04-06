from typing import Optional, Tuple

import torch


class OnnxStreamingEncoder(torch.nn.Module):
    """This class warps the streaming Zipformer to reduce the number of
    state tensors for onnx.
    https://github.com/k2-fsa/icefall/pull/831
    """

    def __init__(self, encoder):
        """
        Args:
            encoder: An instance of Zipformer Class
        """
        super().__init__()
        self.model = encoder

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        len_cache: torch.tensor,
        avg_cache: torch.tensor,
        attn_cache: torch.tensor,
        cnn_cache: torch.tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          len_cache:
            The cached numbers of past frames.
          avg_cache:
            The cached average tensors.
          attn_cache:
            The cached key tensors of the first attention modules.
            The cached value tensors of the first attention modules.
            The cached value tensors of the second attention modules.
          cnn_cache:
            The cached left contexts of the first convolution modules.
            The cached left contexts of the second convolution modules.

        Returns:
          Return a tuple containing 2 tensors:

        """
        num_encoder_layers = []
        encoder_attention_dims = []
        states = []
        for i, encoder in enumerate(self.model.encoders):
            num_encoder_layers.append(encoder.num_layers)
            encoder_attention_dims.append(encoder.attention_dim)

        len_cache = len_cache.transpose(0, 1)  # sum(num_encoder_layers)==15, [15, B]
        offset = 0
        for num_layer in num_encoder_layers:
            states.append(len_cache[offset : offset + num_layer])
            offset += num_layer

        avg_cache = avg_cache.transpose(0, 1)  # [15, B, 384]
        offset = 0
        for num_layer in num_encoder_layers:
            states.append(avg_cache[offset : offset + num_layer])
            offset += num_layer

        attn_cache = attn_cache.transpose(0, 2)  # [15*3, 64, B, 192]
        left_context_len = attn_cache.shape[1]
        offset = 0
        for i, num_layer in enumerate(num_encoder_layers):
            ds = self.model.zipformer_downsampling_factors[i]
            states.append(
                attn_cache[offset : offset + num_layer, : left_context_len // ds]
            )
            offset += num_layer
        for i, num_layer in enumerate(num_encoder_layers):
            encoder_attention_dim = encoder_attention_dims[i]
            ds = self.model.zipformer_downsampling_factors[i]
            states.append(
                attn_cache[
                    offset : offset + num_layer,
                    : left_context_len // ds,
                    :,
                    : encoder_attention_dim // 2,
                ]
            )
            offset += num_layer
        for i, num_layer in enumerate(num_encoder_layers):
            ds = self.model.zipformer_downsampling_factors[i]
            states.append(
                attn_cache[
                    offset : offset + num_layer,
                    : left_context_len // ds,
                    :,
                    : encoder_attention_dim // 2,
                ]
            )
            offset += num_layer

        cnn_cache = cnn_cache.transpose(0, 1)  # [30, B, 384, cnn_kernel-1]
        offset = 0
        for num_layer in num_encoder_layers:
            states.append(cnn_cache[offset : offset + num_layer])
            offset += num_layer
        for num_layer in num_encoder_layers:
            states.append(cnn_cache[offset : offset + num_layer])
            offset += num_layer

        encoder_out, encoder_out_lens, new_states = self.model.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=states,
        )

        new_len_cache = torch.cat(states[: self.model.num_encoders]).transpose(
            0, 1
        )  # [B,15]
        new_avg_cache = torch.cat(
            states[self.model.num_encoders : 2 * self.model.num_encoders]
        ).transpose(
            0, 1
        )  # [B,15,384]
        new_cnn_cache = torch.cat(states[5 * self.model.num_encoders :]).transpose(
            0, 1
        )  # [B,2*15,384,cnn_kernel-1]
        assert len(set(encoder_attention_dims)) == 1
        pad_tensors = [
            torch.nn.functional.pad(
                tensor,
                (
                    0,
                    encoder_attention_dims[0] - tensor.shape[-1],
                    0,
                    0,
                    0,
                    left_context_len - tensor.shape[1],
                    0,
                    0,
                ),
            )
            for tensor in states[
                2 * self.model.num_encoders : 5 * self.model.num_encoders
            ]
        ]
        new_attn_cache = torch.cat(pad_tensors).transpose(0, 2)  # [B,64,15*3,192]

        return (
            encoder_out,
            encoder_out_lens,
            new_len_cache,
            new_avg_cache,
            new_attn_cache,
            new_cnn_cache,
        )


class TritonOnnxDecoder(torch.nn.Module):
    """This class warps the Decoder in decoder.py
    to remove the scalar input "need_pad".
    Triton currently doesn't support scalar input.
    https://github.com/triton-inference-server/server/issues/2333
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
    ):
        """
        Args:
          decoder: A instance of Decoder
        """
        super().__init__()
        self.model = decoder

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        # False to not pad the input. Should be False during inference.
        need_pad = False
        return self.model(y, need_pad)


class TritonOnnxJoiner(torch.nn.Module):
    """This class warps the Joiner in joiner.py
    to remove the scalar input "project_input".
    Triton currently doesn't support scalar input.
    https://github.com/triton-inference-server/server/issues/2333
    "project_input" is set to True.
    Triton solutions only need export joiner to a single joiner.onnx.
    """

    def __init__(
        self,
        joiner: torch.nn.Module,
    ):
        super().__init__()
        self.model = joiner

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
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        # Apply input projections encoder_proj and decoder_proj.
        project_input = True
        return self.model(encoder_out, decoder_out, project_input)
