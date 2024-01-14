# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from subsampling import Conv2dSubsampling, VggSubsampling

try:
    from torchaudio.models import Emformer as _Emformer
except ImportError:
    import torchaudio

    print(
        "Please install torchaudio >= 0.11.0. "
        f"Current version: {torchaudio.__version__}"
    )
    raise


def unstack_states(
    states: List[List[torch.Tensor]],
) -> List[List[List[torch.Tensor]]]:
    """Unstack the emformer state corresponding to a batch of utterances
    into a list of states, were the i-th entry is the state from the i-th
    utterance in the batch.

    Args:
      states:
        A list-of-list of tensors. ``len(states)`` equals to number of
        layers in the emformer. ``states[i]]`` contains the states for
        the i-th layer. ``states[i][k]`` is either a 3-D tensor of shape
        ``(T, N, C)`` or a 2-D tensor of shape ``(C, N)``
    """
    batch_size = states[0][0].size(1)
    num_layers = len(states)

    ans = [None] * batch_size
    for i in range(batch_size):
        ans[i] = [[] for _ in range(num_layers)]

    for li, layer in enumerate(states):
        for s in layer:
            s_list = s.unbind(dim=1)
            # We will use stack(dim=1) later in stack_states()
            for bi, b in enumerate(ans):
                b[li].append(s_list[bi])
    return ans


def stack_states(
    state_list: List[List[List[torch.Tensor]]],
) -> List[List[torch.Tensor]]:
    """Stack list of emformer states that correspond to separate utterances
    into a single emformer state so that it can be used as an input for
    emformer when those utterances are formed into a batch.

    Note:
      It is the inverse of :func:`unstack_states`.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the emformer model for a single utterance.
    Returns:
      Return a new state corresponding to a batch of utterances.
      See the input argument of :func:`unstack_states` for the meaning
      of the returned tensor.
    """
    batch_size = len(state_list)
    ans = []
    for layer in state_list[0]:
        # layer is a list of tensors
        if batch_size > 1:
            ans.append([[s] for s in layer])
            # Note: We will stack ans[layer][s][] later to get ans[layer][s]
        else:
            ans.append([s.unsqueeze(1) for s in layer])

    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states):
            for si, s in enumerate(layer):
                ans[li][si].append(s)
                if b == batch_size - 1:
                    ans[li][si] = torch.stack(ans[li][si], dim=1)
                    # We will use unbind(dim=1) later in unstack_states()
    return ans


class Emformer(EncoderInterface):
    """This is just a simple wrapper around torchaudio.models.Emformer.
    We may replace it with our own implementation some time later.
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        segment_length: int,
        left_context_length: int,
        right_context_length: int,
        max_memory_size: int = 0,
        dropout: float = 0.1,
        subsampling_factor: int = 4,
        vgg_frontend: bool = False,
    ) -> None:
        """
        Args:
          num_features:
            The input dimension of the model.
          output_dim:
            The output dimension of the model.
          d_model:
            Attention dimension.
          nhead:
            Number of heads in multi-head attention.
          dim_feedforward:
            The output dimension of the feedforward layers in encoder.
          num_encoder_layers:
            Number of encoder layers.
          segment_length:
            Number of frames per segment before subsampling.
          left_context_length:
            Number of frames in the left context before subsampling.
          right_context_length:
            Number of frames in the right context before subsampling.
          max_memory_size:
            TODO.
          dropout:
            Dropout in encoder.
          subsampling_factor:
            Number of output frames is num_in_frames // subsampling_factor.
            Currently, subsampling_factor MUST be 4.
          vgg_frontend:
            True to use vgg style frontend for subsampling.
        """
        super().__init__()

        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        if vgg_frontend:
            self.encoder_embed = VggSubsampling(num_features, d_model)
        else:
            self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.segment_length = segment_length  # before subsampling
        self.right_context_length = right_context_length

        assert right_context_length % subsampling_factor == 0
        assert segment_length % subsampling_factor == 0
        assert left_context_length % subsampling_factor == 0

        left_context_length = left_context_length // subsampling_factor
        right_context_length = right_context_length // subsampling_factor
        segment_length = segment_length // subsampling_factor

        self.model = _Emformer(
            input_dim=d_model,
            num_heads=nhead,
            ffn_dim=dim_feedforward,
            num_layers=num_encoder_layers,
            segment_length=segment_length,
            dropout=dropout,
            activation="relu",
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=max_memory_size,
            weight_init_scale_strategy="depthwise",
            tanh_on_mem=False,
            negative_inf=-1e8,
        )

        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(d_model, output_dim)
        )
        self.log_eps = math.log(1e-10)

        self._has_init_state = False
        self._init_state = torch.jit.Attribute([], List[List[torch.Tensor]])

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            Input features of shape (N, T, C).
          x_lens:
            A int32 tensor of shape (N,) containing valid frames in `x` before
            padding. We have `x.size(1) == x_lens.max()`
        Returns:
          Return a tuple containing two tensors:

            - encoder_out, a tensor of shape (N, T', C)
            - encoder_out_lens, a int32 tensor of shape (N,) containing the
              valid frames in `encoder_out` before padding
        """
        x = nn.functional.pad(
            x,
            # (left, right, top, bottom)
            # left/right are for the channel dimension, i.e., axis 2
            # top/bottom are for the time dimension, i.e., axis 1
            (0, 0, 0, self.right_context_length),
            value=self.log_eps,
        )  # (N, T, C) -> (N, T+right_context_length, C)

        x = self.encoder_embed(x)

        # Caution: We assume the subsampling factor is 4!
        x_lens = (((x_lens - 1) >> 1) - 1) >> 1

        emformer_out, emformer_out_lens = self.model(x, x_lens)
        logits = self.encoder_output_layer(emformer_out)

        return logits, emformer_out_lens

    @torch.jit.export
    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
    ):
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C). Note: x also contains right
            context frames.
          x_lens:
            A 2-D tensor of shap containing the number of valid frames for each
            element in `x` before padding. Note: It also counts right context
            frames.
          states:
            Internal states of the model.
        Returns:
          Return a tuple containing 3 tensors:
            - encoder_out, a 3-D tensor of shape (N, T, C)
            - encoder_out_lens: a 1-D tensor of shape (N,)
            - next_state, internal model states for the next invocation
        """
        x = self.encoder_embed(x)

        # Caution: We assume the subsampling factor is 4!
        x_lens = (((x_lens - 1) >> 1) - 1) >> 1

        emformer_out, emformer_out_lens, states = self.model.infer(x, x_lens, states)

        if x.size(1) != (self.model.segment_length + self.model.right_context_length):
            raise ValueError(
                "Incorrect input shape."
                f"{x.size(1)} vs {self.model.segment_length} + "
                f"{self.model.right_context_length}"
            )

        logits = self.encoder_output_layer(emformer_out)

        return logits, emformer_out_lens, states

    @torch.jit.export
    def get_init_state(self, device: torch.device) -> List[List[torch.Tensor]]:
        """Return the initial state of each layer.

        Returns:
          Return the initial state of each layer. NOTE: the returned
          tensors are on the given device. `len(ans) == num_emformer_layers`.
        """
        if self._has_init_state:
            # Note(fangjun): It is OK to share the init state as it is
            # not going to be modified by the model
            return self._init_state

        batch_size = 1

        ans: List[List[torch.Tensor]] = []
        for layer in self.model.emformer_layers:
            s = layer._init_state(batch_size=batch_size, device=device)
            ans.append(s)

        self._has_init_state = True
        self._init_state = ans

        return ans
