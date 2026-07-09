# Copyright 2025 Nanjie Li (linanjie0820@gmail.com)
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
from torch import Tensor


class MoEAdapterDense(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        hidden_mult: float = 1.3,
        num_tasks: int = 2,
        num_langs: int = 0,
        *,
        num_src_langs: int = 0,
        num_tgt_langs: int = 0,
        dropout: float = 0.1,
        router_mid_ratio: float = 0.5,
        entropy_reg: float = 0.0,
        temperature: float = 1.0,
        init_identity: bool = True,
        topk_infer: Optional[int] = None,
        in_dropout: float = 0.0,
    ):
        super().__init__()
        assert temperature > 0.0, "temperature must be > 0"
        assert num_experts >= 1, "num_experts must be >= 1"

        self.d_model = d_model
        self.num_experts = num_experts
        self.entropy_reg = float(entropy_reg)
        self.temperature = float(temperature)
        self.topk_infer = topk_infer

        # --- Experts: MLP(d -> d_hidden -> d) + residual outside ---
        d_hidden = max(1, int(round(d_model * hidden_mult)))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_hidden),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        if init_identity:
            for e in self.experts:
                nn.init.zeros_(e[-1].weight)
                if e[-1].bias is not None:
                    nn.init.zeros_(e[-1].bias)

        # --- Router and conditional embeddings ---
        router_hidden = max(16, int(round(d_model * router_mid_ratio)))
        self.task_embed = nn.Embedding(num_tasks, d_model) if num_tasks > 0 else None
        self.lang_embed = nn.Embedding(num_langs, d_model) if num_langs > 0 else None
        if num_src_langs > 0 or num_tgt_langs > 0:
            assert num_langs == 0, "num_langs must be 0 when using src+tgt embeds"
        self.lang_embed_src = (
            nn.Embedding(num_src_langs, d_model) if num_src_langs > 0 else None
        )
        self.lang_embed_tgt = (
            nn.Embedding(num_tgt_langs, d_model) if num_tgt_langs > 0 else None
        )

        router_in_dim = (
            d_model
            + (d_model if self.task_embed is not None else 0)
            + (d_model if self.lang_embed is not None else 0)
            + (d_model if self.lang_embed_src is not None else 0)
            + (d_model if self.lang_embed_tgt is not None else 0)
        )

        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(router_hidden, num_experts),
        )

        self.in_drop = nn.Dropout(in_dropout)
        self.last_router_weights: Optional[Tensor] = None

    # ---- Helpers -----------------------------------------------------------------
    def _apply_topk(self, w: Tensor) -> Tensor:
        if (self.topk_infer is None) or self.training:
            return w
        k = min(self.topk_infer, w.size(-1))
        topv, topi = torch.topk(w, k, dim=-1)
        mask = torch.zeros_like(w).scatter_(-1, topi, topv)
        return mask / (mask.sum(-1, keepdim=True) + 1e-9)

    # ---- Forward -----------------------------------------------------------------
    def forward(
        self,
        x_tbc: Tensor,
        task_id: Optional[int] = None,
        lang_ids: Optional[Tensor] = None,
        src_lang_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        T, B, C = x_tbc.shape
        assert (
            C == self.d_model
        ), f"Mismatched channel: got {C}, expected {self.d_model}"
        dtype = x_tbc.dtype
        device = x_tbc.device

        conds = [x_tbc]

        if self.task_embed is not None:
            assert task_id is not None, "task_id must be provided when num_tasks > 0"
            t_id = torch.tensor(int(task_id), device=device, dtype=torch.long)
            tvec = self.task_embed(t_id).view(1, 1, C).expand(T, B, C).to(dtype)
            conds.append(tvec)

        if self.lang_embed is not None:
            if lang_ids is None:
                raise ValueError("lang_ids must be provided when num_langs > 0")
            if lang_ids.dtype != torch.long:
                lang_ids = lang_ids.long()
            lvec = (
                self.lang_embed(lang_ids.to(device))
                .view(1, B, C)
                .expand(T, B, C)
                .to(dtype)
            )
            conds.append(lvec)
        if self.lang_embed_src is not None or self.lang_embed_tgt is not None:
            if self.lang_embed_src is not None:
                if src_lang_ids is None:
                    raise ValueError(
                        "src_lang_ids must be provided when src+tgt embeds are enabled"
                    )
                sids = src_lang_ids.to(device)
                if sids.dtype != torch.long:
                    sids = sids.long()
                svec = self.lang_embed_src(sids).view(1, B, C).expand(T, B, C).to(dtype)
                conds.append(svec)
            if self.lang_embed_tgt is not None:
                if lang_ids is None:
                    raise ValueError(
                        "lang_ids (tgt) must be provided when src+tgt embeds are enabled"
                    )
                tids = lang_ids.to(device)
                if tids.dtype != torch.long:
                    tids = tids.long()
                tvec = self.lang_embed_tgt(tids).view(1, B, C).expand(T, B, C).to(dtype)
                conds.append(tvec)

        r_in = torch.cat(conds, dim=-1)  # [T, B, Cin]
        logits = self.router(r_in.float()) / self.temperature
        w = torch.softmax(logits, dim=-1).to(dtype)  # [T, B, E]
        w = self._apply_topk(w)
        self.last_router_weights = w

        x_in = self.in_drop(x_tbc)
        E = torch.stack([e(x_in) for e in self.experts], dim=-1)
        # y = sum_e w_e * expert_e(x)
        y = torch.einsum("tbce,tbe->tbc", E, w)
        return x_tbc + y, w

    # ---- Regularization -----------------------------------------------------------
    def router_entropy_loss(
        self,
        w: Optional[Tensor] = None,
        pad_mask_tb: Optional[Tensor] = None,
    ) -> Tensor:
        if self.entropy_reg <= 0.0:
            p = next(self.parameters())
            return torch.zeros((), device=p.device, dtype=p.dtype)

        if w is None:
            w = self.last_router_weights
        if w is None:
            p = next(self.parameters())
            return torch.zeros((), device=p.device, dtype=p.dtype)

        w_safe = w.clamp_min(1e-9)
        ent_tb = -(w * w_safe.log()).sum(dim=-1)  # [T,B]

        if pad_mask_tb is not None:
            if pad_mask_tb.dtype != torch.bool:
                pad_mask_tb = pad_mask_tb.bool()
            if pad_mask_tb.device != ent_tb.device:
                pad_mask_tb = pad_mask_tb.to(ent_tb.device)
            valid = (~pad_mask_tb).float()
            denom = valid.sum().clamp_min(1.0)
            ent = (ent_tb * valid).sum() / denom
        else:
            ent = ent_tb.mean()
        return -self.entropy_reg * ent
