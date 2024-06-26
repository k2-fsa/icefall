# Copyright    2024     Xiaomi Corp.        (authors: Wei Kang)
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

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn

from icefall import ContextGraph
import logging
import random


class TCPGen(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        embed_dim: int,
        joiner_dim: int,
        decoder: nn.Module,
        attn_dim: int = 512,
        tcpgen_dropout: float = 0.1,
    ):
        super().__init__()
        # embedding for ool (out of biasing list) token
        self.oolemb = torch.nn.Embedding(1, embed_dim)
        # project symbol embeddings
        self.q_proj_sym = torch.nn.Linear(embed_dim, attn_dim)
        # project encoder embeddings
        self.q_proj_acoustic = torch.nn.Linear(encoder_dim, attn_dim)
        # project symbol embeddings (vocabulary + ool)
        self.k_proj = torch.nn.Linear(embed_dim, attn_dim)
        # generate tcpgen probability
        self.tcp_gate = torch.nn.Linear(attn_dim + joiner_dim, 1)
        self.dropout_tcpgen = torch.nn.Dropout(tcpgen_dropout)
        self.decoder = decoder
        self.vocab_size = decoder.vocab_size

    def get_tcpgen_masks(
        self,
        targets: torch.Tensor,
        context_graph: ContextGraph,
        vocab_size: int,
        blank_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sql_len = targets.shape
        dist_masks = torch.ones((batch_size, sql_len, vocab_size + 1))
        gen_masks = []
        yseqs = targets.tolist()
        for i, yseq in enumerate(yseqs):
            node = context_graph.root
            gen_mask = []
            for j, y in enumerate(yseq):
                not_matched = False
                if y == blank_id:
                    node = context_graph.root
                    gen_mask.append(0)
                elif y in node.next:
                    gen_mask.append(0)
                    node = node.next[y]
                    if node.is_end:
                        node = context_graph.root
                else:
                    gen_mask.append(1)
                    node = context_graph.root
                    not_matched = True
                # unmask_index = (
                #    [vocab_size]
                #    if node.token == -1
                #    else list(node.next.keys()) + [vocab_size]
                # )
                # logging.info(f"token : {node.token}, keys : {node.next.keys()}")
                # dist_masks[i, j, unmask_index] = 0
                if not not_matched:
                    dist_masks[i, j, list(node.next.keys())] = 0

            gen_masks.append(gen_mask + [1] * (sql_len - len(gen_mask)))
        gen_masks = torch.Tensor(gen_masks).to(targets.device).bool()
        dist_masks = dist_masks.to(targets.device).bool()
        if random.random() >= 0.95:
            logging.info(
                f"gen_mask nonzero {gen_masks.shape} : {torch.count_nonzero(torch.logical_not(gen_masks), dim=1)}"
            )
            logging.info(
                f"dist_masks nonzero {dist_masks.shape} : {torch.count_nonzero(torch.logical_not(dist_masks), dim=2)}"
            )
        return dist_masks, gen_masks

    def get_tcpgen_distribution(
        self, query: torch.Tensor, dist_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          query:
            shape : (B, T, s_range, attn_dim)
          dist_masks:
            shape : (B, T, s_range, V + 1)
        """
        # From original paper, k, v share same embeddings
        # (V + 1, embed_dim)
        kv = torch.cat([self.decoder.embedding.weight.data, self.oolemb.weight], dim=0)
        # (V + 1, attn_dim)
        kv = self.dropout_tcpgen(self.k_proj(kv))
        # (B, T, s_range, attn_dim) * (attn_dim, V + 1) -> (B, T, s_range, V + 1)
        distribution = torch.matmul(query, kv.permute(1, 0)) / math.sqrt(query.size(-1))
        distribution = distribution.masked_fill(
            dist_masks, torch.finfo(distribution.dtype).min
        )
        distribution = distribution.softmax(dim=-1)
        # (B, T, s_range, V) * (V, attn_dim) -> (B, T, s_range, attn_dim)
        # logging.info(f"distribution shape : {distribution.shape}")
        hptr = torch.matmul(distribution[:, :, :, :-1], kv[:-1, :])
        hptr = self.dropout_tcpgen(hptr)
        if random.random() > 0.95:
            logging.info(
                f"distribution mean : {torch.mean(distribution, dim=3)}, std: {torch.std(distribution, dim=3)}"
            )
            logging.info(
                f"distribution min : {torch.min(distribution, dim=3)}, max: {torch.max(distribution, dim=3)}"
            )
        return hptr, distribution

    def prune_query_and_mask(
        self,
        query_sym: torch.Tensor,
        query_acoustic: torch.Tensor,
        dist_masks: torch.Tensor,
        gen_masks: torch.Tensor,
        ranges: torch.Tensor,
    ):
        """Prune the queries from symbols and acoustics with ranges
        generated by `get_rnnt_prune_ranges` in pruned rnnt loss.

        Args:
          query_sym:
            The symbol query, with shape (B, S, attn_dim).
          query_acoustic:
            The acoustic query, with shape (B, T, attn_dim).
          dist_masks:
            The TCPGen distribution masks, with shape (B, S, V + 1).
          gen_masks:
            The TCPGen probability masks, with shape (B, S).
          ranges:
            A tensor containing the symbol indexes for each frame that we want to
            keep. Its shape is (B, T, s_range), see the docs in
            `get_rnnt_prune_ranges` in rnnt_loss.py for more details of this tensor.

        Returns:
          Return the pruned query with the shape (B, T, s_range, attn_dim).
        """
        assert ranges.shape[0] == query_sym.shape[0], (ranges.shape, query_sym.shape)
        assert ranges.shape[0] == query_acoustic.shape[0], (
            ranges.shape,
            query_acoustic.shape,
        )
        assert query_acoustic.shape[1] == ranges.shape[1], (
            query_acoustic.shape,
            ranges.shape,
        )
        (B, T, s_range) = ranges.shape
        (B, S, attn_dim) = query_sym.shape
        assert query_acoustic.shape == (B, T, attn_dim), (
            query_acoustic.shape,
            (B, T, attn_dim),
        )
        assert dist_masks.shape == (B, S, self.vocab_size + 1), (
            dist_masks.shape,
            (B, S, self.vocab_size + 1),
        )
        assert gen_masks.shape == (B, S), (gen_masks.shape, (B, S))
        # (B, T, s_range, attn_dim)
        query_acoustic_pruned = query_acoustic.unsqueeze(2).expand(
            (B, T, s_range, attn_dim)
        )
        # logging.info(f"query_sym : {query_sym.shape}")
        # logging.info(f"ranges : {ranges}")
        # (B, T, s_range, attn_dim)
        query_sym_pruned = torch.gather(
            query_sym,
            dim=1,
            index=ranges.reshape(B, T * s_range, 1).expand((B, T * s_range, attn_dim)),
        ).reshape(B, T, s_range, attn_dim)
        # (B, T, s_range, V + 1)
        dist_masks_pruned = torch.gather(
            dist_masks,
            dim=1,
            index=ranges.reshape(B, T * s_range, 1).expand(
                (B, T * s_range, self.vocab_size + 1)
            ),
        ).reshape(B, T, s_range, self.vocab_size + 1)
        # (B, T, s_range)
        gen_masks_pruned = torch.gather(
            gen_masks, dim=1, index=ranges.reshape(B, T * s_range)
        ).reshape(B, T, s_range)
        return (
            query_sym_pruned + query_acoustic_pruned,
            dist_masks_pruned,
            gen_masks_pruned,
        )

    def forward(
        self,
        targets: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        ranges: torch.Tensor,
        context_graph: ContextGraph,
    ):
        """
        Args:
          target:
            The training targets in token ids (padded with blanks). shape : (B, S)
          encoder_embeddings:
            The encoder outputs. shape: (B, T, attn_dim)
          ranges:
            The prune ranges from pruned rnnt. shape: (B, T, s_range)
          context_graphs:
            The context_graphs for each utterance. B == len(context_graphs).

        Return:
          returns tcpgen embedding with shape (B, T, s_range, attn_dim) and
          tcpgen distribution with shape (B, T, s_range, V + 1).
        """
        query_sym = self.decoder.embedding(targets)

        query_sym = self.q_proj_sym(query_sym)  # (B, S, attn_dim)
        query_acoustic = self.q_proj_acoustic(encoder_embeddings)  # (B , T, attn_dim)

        # dist_masks : (B, S, V + 1)
        # gen_masks : (B, S)
        dist_masks, gen_masks = self.get_tcpgen_masks(
            targets=targets, context_graph=context_graph, vocab_size=self.vocab_size
        )
        # query : (B, T, s_range, attn_dim)
        # dist_masks : (B, T, s_range, V + 1)
        query, dist_masks, gen_masks = self.prune_query_and_mask(
            query_sym=query_sym,
            query_acoustic=query_acoustic,
            dist_masks=dist_masks,
            gen_masks=gen_masks,
            ranges=ranges,
        )

        if random.random() >= 0.95:
            logging.info(
                f"pruned gen_mask nonzero {gen_masks.shape} : {torch.count_nonzero(torch.logical_not(gen_masks), dim=1)}"
            )
            logging.info(
                f"pruned dist_masks nonzero {dist_masks.shape} : {torch.count_nonzero(torch.logical_not(dist_masks), dim=3)}"
            )

        # hptr : (B, T, s_range, attn_dim)
        # tcpgen_dist : (B, T, s_range, V + 1)
        hptr, tcpgen_dist = self.get_tcpgen_distribution(query, dist_masks)
        return hptr, tcpgen_dist, gen_masks

    def generator_prob(
        self, hptr: torch.Tensor, h_joiner: torch.Tensor, gen_masks: torch.Tensor
    ) -> torch.Tensor:
        # tcpgen_prob : (B, T, s_range, 1)
        tcpgen_prob = self.tcp_gate(torch.cat((h_joiner, hptr), dim=-1))
        tcpgen_prob = torch.sigmoid(tcpgen_prob)
        tcpgen_prob = tcpgen_prob.masked_fill(gen_masks.unsqueeze(-1), 0)
        if random.random() >= 0.95:
            logging.info(
                f"tcpgen_prob mean : {torch.mean(tcpgen_prob.squeeze(-1), dim=(1,2))}, std : {torch.std(tcpgen_prob.squeeze(-1), dim=(1, 2))}"
            )
            logging.info(
                f"tcpgen_prob min : {torch.min(tcpgen_prob.squeeze(-1), dim=1)}, max : {torch.max(tcpgen_prob.squeeze(-1), dim=1)}"
            )
        return tcpgen_prob
