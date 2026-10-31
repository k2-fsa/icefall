# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from scaling import Balancer


class Decoder(nn.Module):
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

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
        )
        # the balancers are to avoid any drift in the magnitude of the
        # embeddings, which would interact badly with parameter averaging.
        self.balancer = Balancer(
            decoder_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size

        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim // 4,  # group size == 4
                bias=False,
            )
            self.balancer2 = Balancer(
                decoder_dim,
                channel_dim=-1,
                min_positive=0.0,
                max_positive=1.0,
                min_abs=0.5,
                max_abs=1.0,
                prob=0.05,
            )
        else:
            # To avoid `RuntimeError: Module 'Decoder' has no attribute 'conv'`
            # when inference with torch.jit.script and context_size == 1
            self.conv = nn.Identity()
            self.balancer2 = nn.Identity()

    def forward(self, y: torch.Tensor, need_pad: bool = True) -> torch.Tensor:
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
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        embedding_out = self.balancer(embedding_out)

        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
            embedding_out = F.relu(embedding_out)
            embedding_out = self.balancer2(embedding_out)

        return embedding_out


class GroupedConv1dAsMatMul(nn.Module):
    """Inference-only replacement for the decoder's grouped Conv1d.

    During inference (need_pad=False) the input of the decoder conv has
    exactly kernel_size frames, so the conv produces a single output frame.
    A grouped conv with one output frame is a block-diagonal linear map of
    the flattened input, so it can be computed exactly as one MatMul:

        y[n, co, 0] = sum_{i, k} W[co, i, k] * x[n, g * cin_g + i, k],
        g = co // cout_g

    which equals flatten(x)[n, :] @ W_dense[:, co] with
    W_dense[(g * cin_g + i) * kernel_size + k, co] = W[co, i, k].

    When exported to ONNX this gives Reshape -> MatMul -> Unsqueeze instead
    of a Conv node. ONNX Runtime's CUDA Conv kernel holds a per-node mutex
    while it (re)builds its cuDNN plan for every new input shape, so a
    decoder session shared by several threads with varying batch sizes is
    serialized on that Conv; the MatMul form has no such lock.
    """

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        assert conv.bias is None, "only bias-free convs are supported"
        assert conv.padding in (0, (0,)), conv.padding
        assert conv.stride in (1, (1,)), conv.stride
        assert conv.dilation in (1, (1,)), conv.dilation

        # (out_channels, in_channels // groups, kernel_size)
        weight = conv.weight.detach()
        out_channels, cin_g, kernel_size = weight.shape
        groups = conv.groups
        cout_g = out_channels // groups

        blocks = []
        for g in range(groups):
            w = weight[g * cout_g : (g + 1) * cout_g]
            blocks.append(w.reshape(cout_g, cin_g * kernel_size).t())
        # (in_channels * kernel_size, out_channels), block diagonal
        self.register_buffer("weight", torch.block_diag(*blocks).contiguous())
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, in_channels, kernel_size). Decoder.forward
            checks the length when need_pad is False.
        Returns:
          Return a 3-D tensor of shape (N, out_channels, 1).
        """
        x = torch.flatten(x, start_dim=1)
        return torch.matmul(x, self.weight).unsqueeze(-1)


def replace_decoder_conv_with_matmul(decoder: Decoder) -> None:
    """Replace decoder.conv in-place by an equivalent GroupedConv1dAsMatMul.

    Only valid for inference with need_pad=False, e.g., before ONNX export.
    Does nothing if the decoder has no conv (context_size == 1).
    """
    if isinstance(decoder.conv, nn.Conv1d):
        decoder.conv = GroupedConv1dAsMatMul(decoder.conv)
