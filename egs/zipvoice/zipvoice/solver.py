#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
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

from typing import Optional, Union

import torch


class DiffusionModel(torch.nn.Module):
    """A wrapper of diffusion models for inference.
    Args:
        model: The diffusion model.
        distill: Whether it is a distillation model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__()
        self.model = model
        self.distill = distill
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that Handles the classifier-free guidance.
        Args:
            t: The current timestep, a tensor of shape (batch, 1, 1) or a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            speech_condition: The speech_condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )
        if self.distill:
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )

        if (guidance_scale == 0.0).all():
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                **kwargs
            )
        else:
            if t.dim() != 0:
                t = torch.cat([t] * 2, dim=0)

            x = torch.cat([x] * 2, dim=0)
            padding_mask = torch.cat([padding_mask] * 2, dim=0)

            text_condition = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )

            if t.dim() == 0:
                if t > 0.5:
                    speech_condition = torch.cat(
                        [torch.zeros_like(speech_condition), speech_condition], dim=0
                    )
                else:
                    guidance_scale = guidance_scale * 2
                    speech_condition = torch.cat(
                        [speech_condition, speech_condition], dim=0
                    )
            else:
                assert t.dim() > 0, t
                larger_t_index = (t > 0.5).squeeze(1).squeeze(1)
                zero_speech_condition = torch.cat(
                    [torch.zeros_like(speech_condition), speech_condition], dim=0
                )
                speech_condition = torch.cat(
                    [speech_condition, speech_condition], dim=0
                )
                speech_condition[larger_t_index] = zero_speech_condition[larger_t_index]
                guidance_scale[~larger_t_index[: larger_t_index.size(0) // 2]] *= 2

            data_uncond, data_cond = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                **kwargs
            ).chunk(2, dim=0)

            res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return res


class EulerSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_fm_decoder",
    ):
        """Construct a Euler Solver
        Args:
            model: The diffusion model.
            distill: Whether it is distillation model.
        """

        self.model = DiffusionModel(model, distill=distill, func_name=func_name)

    def sample(
        self,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: torch.Tensor,
        num_step: int = 10,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        t_start: Union[float, torch.Tensor] = 0.0,
        t_end: Union[float, torch.Tensor] = 1.0,
        t_shift: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the sample at time `t_end` by Euler Solver.
        Args:
            x: The initial value at time `t_start`, with the shape (batch, seq_len, emb_dim).
            text_condition: The text condition of the diffision mode, with the shape (batch, seq_len, emb_dim).
            speech_condition: The speech condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the shape (batch, seq_len).
            num_step: The number of ODE steps.
            guidance_scale: The scale for classifier-free guidance, which is
                a float or a tensor with the shape (batch, 1, 1).
            t_start: the start timestep in the range of [0, 1],
                which is a float or a tensor with the shape (batch, 1, 1).
            t_end: the end time_step in the range of [0, 1],
                which is a float or a tensor with the shape (batch, 1, 1).
            t_shift: shift the t toward smaller numbers so that the sampling
                will emphasize low SNR region. Should be in the range of (0, 1].
                The shifting will be more significant when the number is smaller.

        Returns:
            The approximated solution at time `t_end`.
        """
        device = x.device

        if torch.is_tensor(t_start) and t_start.dim() > 0:
            timesteps = get_time_steps_batch(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        else:
            timesteps = get_time_steps(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        for step in range(num_step):
            v = self.model(
                t=timesteps[step],
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )
            x = x + v * (timesteps[step + 1] - timesteps[step])
        return x


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling.

    Args:
        t_start: The starting time of the sampling (default is 0).
        t_end: The starting time of the sampling (default is 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1,).
    """

    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps


def get_time_steps_batch(
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling in the batch mode.

    Args:
        t_start: The starting time of the sampling (default is 0), with the shape (batch, 1, 1).
        t_end: The starting time of the sampling (default is 1), with the shape (batch, 1, 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1, N, 1, 1).
    """
    while t_start.dim() > 1 and t_start.size(-1) == 1:
        t_start = t_start.squeeze(-1)
    while t_end.dim() > 1 and t_end.size(-1) == 1:
        t_end = t_end.squeeze(-1)
    assert t_start.dim() == t_end.dim() == 1

    timesteps_shape = (num_step + 1, t_start.size(0))
    timesteps = torch.zeros(timesteps_shape, device=device)

    for i in range(t_start.size(0)):
        timesteps[:, i] = torch.linspace(t_start[i], t_end[i], steps=num_step + 1)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps.unsqueeze(-1).unsqueeze(-1)
