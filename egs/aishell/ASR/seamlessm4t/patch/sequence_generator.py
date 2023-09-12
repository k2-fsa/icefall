# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from fairseq2.data import Collater, SequenceData, VocabularyInfo
from fairseq2.generation.beam_search import BeamSearch, StandardBeamSearch
from fairseq2.generation.logits_processor import LogitsProcessor
from fairseq2.models.encoder_decoder import Seq2SeqDecoder
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import Device


@dataclass
class SequenceGeneratorOptions:
    """Holds the options to pass to a sequence generator."""

    beam_size: int = 5
    """The beam size."""

    min_seq_len: int = 1
    """The minimum length of generated sequences (including prefix sequence)."""

    soft_max_seq_len: Optional[Tuple[int, int]] = (1, 200)
    """The terms ``a`` and ``b`` of ``ax + b`` where ``x`` is the source
    sequence length. The generated sequences (including prefix sequence) will
    have the maximum length of ``min(hard_max_seq_len, ax + b)``. See also
    ``hard_max_seq_len``."""

    hard_max_seq_len: int = 1024
    """The hard limit on maximum length of generated sequences."""

    len_penalty: float = 1.0
    """The length penalty, where values less than 1.0 favor shorter, values
    greater than 1.0 favor longer sequences."""

    unk_penalty: float = 0.0
    """The unknown symbol penalty, where values less than 0 produce more UNKs,
    values greater than 0 produce fewer UNKs."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by the length of generated sequences."""

    search: Optional[BeamSearch] = None
    """The beam search algorithm to use."""

    logits_processor: Optional[LogitsProcessor] = None
    """Logits processor called before applying beam search step."""


class Seq2SeqGenerator:
    """Represents a sequence-to-sequence generator."""

    decoder: Seq2SeqDecoder
    opts: SequenceGeneratorOptions
    beam_size: int
    eos_idx: int
    pad_idx: Optional[int]
    unk_idx: Optional[int]
    prefix_seq: Union[int, Tensor]
    prefix_seq_len: int
    search: BeamSearch
    logits_processor: Optional[LogitsProcessor]
    collater: Collater

    def __init__(
        self,
        decoder: Seq2SeqDecoder,
        vocab_info: VocabularyInfo,
        prefix_seq: Optional[Union[int, Tensor]],
        opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param decoder:
            The decoder to use.
        :param vocab_info:
            The vocabulary information to use.
        :param prefix_seq:
            The prefix sequence, typically one or more control symbols
            indicating the beginning of a sequence. *Shape:* :math:`()` or
            :math:`(S)`, where :math:`S` is the sequence length. If ``None``,
            the EOS symbol will be used as prefix.
        :param opts:
            The generation options.
        """
        self.decoder = decoder

        self.opts = opts or SequenceGeneratorOptions()

        # Set beam size.
        if vocab_info.pad_idx is None:
            self.beam_size = min(self.opts.beam_size, vocab_info.size)
        else:
            # -1 since we never select PAD.
            self.beam_size = min(self.opts.beam_size, vocab_info.size - 1)

        if vocab_info.eos_idx is None:
            raise ValueError(
                "`vocab_info` must have `eos_idx` set for sequence generation."
            )

        # Set vocab info.
        self.eos_idx = 1
        #self.eos_idx = vocab_info.eos_idx
        self.unk_idx = 2
        #self.unk_idx = vocab_info.unk_idx
        self.pad_idx = 0
        #self.pad_idx = vocab_info.pad_idx

        # Set prefix sequence.
        if 1:
        #if prefix_seq is None:
            # If `None`, we follow fairseq's convention, and use EOS as the
            # prefix.
            self.prefix_seq, self.prefix_seq_len = self.eos_idx, 1
        else:
            self.prefix_seq = prefix_seq

            if isinstance(prefix_seq, Tensor):
                num_dim = prefix_seq.dim()

                if num_dim >= 2:
                    raise ValueError(
                        f"`prefix_seq` must be a scalar or a 1-dimensional tensor, but is {num_dim}-dimensional instead."
                    )

                self.prefix_seq_len = 1 if num_dim == 0 else prefix_seq.size(0)
            else:
                self.prefix_seq_len = 1

        # Set beam search.
        self.search = self.opts.search or StandardBeamSearch()
        self.logits_processor = self.opts.logits_processor

        if vocab_info.pad_idx is None:
            self.collater = Collater()
        else:
            self.collater = Collater(self.pad_idx, pad_to_multiple=2)

    @torch.inference_mode()
    def __call__(
        self,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        source_seq_len: Optional[int] = None,
    ) -> "SequenceGeneratorOutput":
        opts = self.opts

        num_searches = encoder_output.size(0)

        beam_size = opts.beam_size

        max_seq_len = self._determine_max_seq_len(source_seq_len)

        device = encoder_output.device

        encoder_output, encoder_padding_mask = self._fan_out_encoder_output(
            encoder_output, encoder_padding_mask
        )

        # Each element contains the id of the search corresponding to a single
        # source sequence and its hypotheses.
        active_searches: List[Tuple[int, List[Hypothesis]]] = [
            (search_idx, []) for search_idx in range(num_searches)
        ]

        # Once a source sequence has `beam_size` hypotheses, its search is moved
        # from `active_searches` to `finished_searches`.
        finished_searches: List[List[Hypothesis]] = [[] for i in range(num_searches)]

        num_remaining_searches = num_searches

        # Initialize buffers.
        # (N x B, S)
        seqs = torch.zeros(
            (num_searches * beam_size, max_seq_len), device=device, dtype=torch.int64
        )

        # (N x B, S)
        scores = torch.zeros(
            (num_searches * beam_size, max_seq_len), device=device, dtype=torch.float32
        )

        # A list that indicates beams that should be ignored in the next step.
        ignored_beam_mask = torch.full(
            (num_searches, beam_size), False, device=device, dtype=torch.bool
        )

        # An offset array for converting between batch-wide and search-local
        # beam indices.
        # (B)
        search_offsets = torch.arange(num_searches, device=device) * beam_size

        # (B) -> (B, 1)
        search_offsets.unsqueeze_(-1)

        cand_offsets = torch.arange(2 * beam_size, device=device)

        state_bag = IncrementalStateBag()

        # At this point, the state is fully initialized, kick off the search.
        self._bootstrap_seqs_and_scores(
            seqs, scores, encoder_output, encoder_padding_mask, state_bag
        )

        start_step = self.prefix_seq_len - 1

        # Holds the indices of beams (a beam can occur more than once) that we
        # should continue with in the next step.
        beam_indices: Optional[Tensor] = None

        # Holds the indices of searches that we should continue with in the next
        # step. If not `None`, it means we finalized one or more searches in the
        # last step.
        search_indices: Optional[Tensor] = None

        for step_nr in range(start_step, max_seq_len - 1):
            if beam_indices is not None:
                # If not `None`, it means in the last step we finalized one or
                # more searches. We should ensure that we adjust `beam_indices`
                # before reordering `decoder`'s incremental state.
                if search_indices is not None:
                    num_searches = search_indices.numel()

                    # (N)
                    delta = search_indices - torch.arange(num_searches, device=device)

                    # (N) -> (N, 1)
                    delta.unsqueeze_(-1)

                    # Adjust indices to take into account removed searches.
                    beam_indices.view(num_searches, beam_size).add_(delta * beam_size)

                state_bag.reorder(beam_indices)

            decoder_output, decoder_padding_mask = self.decoder.decode(
                seqs[:, step_nr : step_nr + 1],
                None,  # We never generate PAD.
                encoder_output,
                encoder_padding_mask,
                state_bag,
            )

            state_bag.increment_step()

            model_output = self.decoder.project(decoder_output, decoder_padding_mask)

            # lprobs:          (1, V)
            # model_output: (N, 1, V)
            lprobs = log_softmax(model_output.logits, dim=-1, dtype=torch.float32)

            # Do not allow EOS before reaching the minimum sequence length.
            if step_nr < self.opts.min_seq_len:
                lprobs[:, :, self.eos_idx] = -torch.inf

            # fmt: off
            # If we have reached the maximum length, force the last step to be
            # EOS.
            if step_nr == max_seq_len - 2:
                lprobs[:, :, : self.eos_idx]       = -torch.inf
                lprobs[:, :,   self.eos_idx + 1 :] = -torch.inf
            # fmt: on

            # Never allow PAD.
            if self.pad_idx is not None:
                lprobs[:, :, self.pad_idx] = -torch.inf

            # Apply UNK penalty.
            if self.unk_idx is not None:
                lprobs[:, :, self.unk_idx] -= self.opts.unk_penalty

            # update scores in place using logits_processor
            if self.logits_processor is not None:
                self.logits_processor(
                    seqs.view(num_searches, beam_size, -1)[:, :, : step_nr + 1],
                    lprobs.view(num_searches, beam_size, -1),
                )

            # Determine candidates for the next step.
            # (N, 2 x B)
            cand_scores, cand_indices, cand_beam_indices = self.search.step(
                step_nr,
                step_nr == start_step,
                lprobs.view(num_searches, beam_size, -1),
                scores.view(num_searches, beam_size, -1)[:, :, : step_nr + 1],
            )

            # Convert search-local beam indices to batch-wide beam indices.
            # (N, 2 x B) + (N) -> (N, 2 x B)
            global_cand_beam_indices = cand_beam_indices + search_offsets

            # Finalize beams that reached the minimum length and that end with
            # an EOS.
            # (N, 2 x B)
            eos_mask = (cand_indices == self.eos_idx) & (cand_scores != -math.inf)

            # Do not attempt to finalize beams that should be ignored.
            eos_mask[:, :beam_size][ignored_beam_mask] = False

            # Only consider EOS when it's among the top `beam_size` indices. Now
            # we know what beam(s) to finalize.
            # (N, B)
            eos_beam_indices = torch.masked_select(
                global_cand_beam_indices[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            if eos_beam_indices.numel() > 0:
                # Select the scores of the finalized beams.
                # (N, B)
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                newly_finished_searches = self._finalize_hypothesis(
                    step_nr,
                    eos_beam_indices,
                    eos_scores,
                    seqs,
                    scores,
                    active_searches,
                    finished_searches,
                )

                num_remaining_searches -= len(newly_finished_searches)

                if num_remaining_searches == 0:
                    break
            else:
                newly_finished_searches = None

            # Remove finished searches (ones for which `beam_size` finalized
            # beams have been generated) from the batch.
            if newly_finished_searches:
                new_num_searches = num_searches - len(newly_finished_searches)

                # Construct `search_indices` which holds indices of searches
                # to keep for the next step.
                search_mask = torch.full((num_searches,), True, device=device)

                search_mask[newly_finished_searches] = False

                search_indices = torch.arange(num_searches, device=device)

                search_indices = search_indices.masked_select(search_mask)

                # fmt: off
                # Filter out removed batches from state variables.
                # (N, B) -> (N - F, B)
                ignored_beam_mask = ignored_beam_mask[search_indices]

                # (N, 2 x B) -> (N - F, 2 x B)
                cand_scores       = cand_scores      [search_indices]
                cand_indices      = cand_indices     [search_indices]
                cand_beam_indices = cand_beam_indices[search_indices]

                # (N) -> (N - F)
                search_offsets.resize_(new_num_searches, 1)

                # (N - F, 2 x B) + (N - F) -> (N - F, 2 x B)
                global_cand_beam_indices = cand_beam_indices + search_offsets

                # (N, 2 x B) -> (N - F, 2 x B)
                eos_mask = eos_mask[search_indices]

                # (N x B, S) -> (N, B, S)
                seqs   = seqs  .view(num_searches, -1)
                scores = scores.view(num_searches, -1)

                # (N, B, S + 1) -> ((N - F) x B, S)
                seqs   = seqs  [search_indices].view(new_num_searches * beam_size, -1)
                scores = scores[search_indices].view(new_num_searches * beam_size, -1)

                # (N x B, S_enc, M) -> (N, B, S_enc, M)
                encoder_output = encoder_output.unflatten(0, (num_searches, -1))

                # (N, B, S_enc, M) -> ((N - F) x B, S_enc, M)
                encoder_output = encoder_output[search_indices].flatten(0, 1)

                if encoder_padding_mask is not None:
                    # (N x B, S_enc, M) -> (N, B, S_enc, M)
                    padding_mask = encoder_padding_mask.unflatten(0, (num_searches, -1))

                    # (N, B, S_enc, M) -> ((N - F) x B, S_enc, M)
                    encoder_padding_mask = padding_mask[search_indices].flatten(0, 1)
                # fmt: on

                num_searches = new_num_searches
            else:
                search_indices = None

            eos_mask[:, :beam_size][ignored_beam_mask] = True

            # Set `beam_weights` so that values greater than or equal to 2 x
            # `beam_size` indicate finished beams (i.e. end with EOS) and values
            # less than 2 x `beam_size` indicate active beams.
            # (N, 2 x B)
            beam_weights = cand_offsets + (eos_mask * (2 * beam_size))

            # Get the top `beam_size` active beams, which are the beams with the
            # smallest weights in `active_beam_weights`.
            # (N, B)
            active_beam_weights, active_beams = torch.topk(
                beam_weights, k=beam_size, dim=1, largest=False
            )

            # Update to ignore finalized beams in the next step.
            # (N, B)
            ignored_beam_mask = active_beam_weights >= 2 * beam_size

            # We should always have at least one active beam in each search.
            assert (~ignored_beam_mask).any(dim=1).all()

            # Denotes which beams are continued for each new hypothesis (a beam
            # can be selected more than once).
            # (N, B)
            beam_indices = torch.gather(
                global_cand_beam_indices, dim=1, index=active_beams
            )

            # (N, B) -> (N x B)
            beam_indices = beam_indices.view(-1)

            # fmt: off
            # Reorder beams in the `seq` and `score` buffers. The same beam can
            # be selected more than once.
            if step_nr > start_step:
                seqs  [:, : step_nr + 1] = torch.index_select(
                    seqs  [:, : step_nr + 1], dim=0, index=beam_indices
                )
                scores[:, : step_nr + 1] = torch.index_select(
                    scores[:, : step_nr + 1], dim=0, index=beam_indices
                )

            # (N x B, S) -> (N, B, S)
            seqs_view   = seqs  .view(num_searches, beam_size, -1)
            scores_view = scores.view(num_searches, beam_size, -1)

            seqs_view  [:, :, step_nr + 1] = torch.gather(cand_indices, dim=1, index=active_beams)
            scores_view[:, :, step_nr + 1] = torch.gather(cand_scores,  dim=1, index=active_beams)
            # fmt: on

        # Ensure that hypotheses are sorted by their scores before returning.
        for batch in finished_searches:
            batch.sort(key=lambda b: b.score, reverse=True)  # type: ignore[arg-type, return-value]

        return SequenceGeneratorOutput(
            results=finished_searches, device=device, collater=self.collater
        )

    def _determine_max_seq_len(self, source_seq_len: Optional[int]) -> int:
        opts = self.opts

        if source_seq_len is None or opts.soft_max_seq_len is None:
            max_seq_len = opts.hard_max_seq_len
        else:
            at, bt = opts.soft_max_seq_len

            max_seq_len = min(opts.hard_max_seq_len, int(at * source_seq_len + bt))

        if opts.min_seq_len > max_seq_len:
            raise ValueError(
                f"The effective maximum sequence length must be greater than or equal to `min_seq_len` ({opts.min_seq_len}), but is {max_seq_len} instead. Adjust your soft and hard maximum sequence length limits."
            )

        if self.prefix_seq_len >= max_seq_len:
            raise ValueError(
                f"The effective maximum sequence length must be greater than `prefix_seq_len` ({self.prefix_seq_len}), but is {max_seq_len} instead."
            )

        return max_seq_len

    def _fan_out_encoder_output(
        self, encoder_output: Tensor, encoder_padding_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        num_searches = encoder_output.size(0)  # i.e. batch size

        # Fan out `encoder_output` to `num_searches` x `beam_size`.
        # (N)
        fan_out_indices = torch.arange(num_searches, device=encoder_output.device)

        # (N) -> (N x B)
        fan_out_indices = fan_out_indices.repeat_interleave(self.beam_size)

        # (N, S_enc, M) -> (N x B, S_enc, M)
        encoder_output = encoder_output.index_select(dim=0, index=fan_out_indices)

        # (N, S_enc, M) -> (N x B, S_enc, M)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.index_select(
                dim=0, index=fan_out_indices
            )

        return encoder_output, encoder_padding_mask

    def _bootstrap_seqs_and_scores(
        self,
        seqs: Tensor,
        scores: Tensor,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: IncrementalStateBag,
    ) -> None:
        assert self.prefix_seq_len > 0

        seqs[:, : self.prefix_seq_len] = self.prefix_seq

        if self.prefix_seq_len == 1:
            return

        assert isinstance(self.prefix_seq, Tensor)

        # We have to bootstrap the model with the already fanned-out encoder
        # output to correctly initialize its incremental state. This causes some
        # redundancy as we have to expand `decoder_input` to match the shape of
        # `encoder_output`.
        # (S_pfx) -> (N x B, S_pfx - 1)
        decoder_input = self.prefix_seq[:-1].expand(encoder_output.size(0), -1)

        # Bootstrap the model state with prefix sequence.
        decoder_output, decoder_padding_mask = self.decoder.decode(
            decoder_input,
            None,
            encoder_output,
            encoder_padding_mask,
            state_bag,
        )

        state_bag.increment_step(self.prefix_seq_len - 1)

        model_output = self.decoder.project(decoder_output, decoder_padding_mask)

        # lprobs:          (S_pfx - 1, V)
        # model_output: (N, S_pfx - 1, V) -> (S_pfx - 1, V)
        lprobs = log_softmax(model_output.logits[0], dim=-1, dtype=torch.float32)

        # Fetch scores of next steps.
        # (S_pfx - 1, 1)
        prefix_scores = torch.take_along_dim(
            lprobs, indices=self.prefix_seq[1:].unsqueeze(1), dim=-1
        )

        # (S_pfx - 1, 1) -> (S_pfx - 1)
        prefix_scores.squeeze_(1).cumsum_(dim=0)

        # First step (e.g. EOS)'s score is always 0.
        scores[:, 1 : self.prefix_seq_len] = prefix_scores

    def _finalize_hypothesis(
        self,
        step_nr: int,
        eos_beam_indices: Tensor,
        eos_scores: Tensor,
        seqs: Tensor,
        scores: Tensor,
        active_searches: List[Tuple[int, List["Hypothesis"]]],
        finished_searches: List[List["Hypothesis"]],
    ) -> List[int]:
        # fmt: off
        finalized_seqs   = seqs  .index_select(dim=0, index=eos_beam_indices)
        finalized_scores = scores.index_select(dim=0, index=eos_beam_indices)

        finalized_seqs   = finalized_seqs  [:, : step_nr + 2]
        finalized_scores = finalized_scores[:, : step_nr + 2]

        # Finalize beams.
        finalized_seqs  [:, -1] = self.eos_idx
        finalized_scores[:, -1] = eos_scores
        # fmt: on

        # Convert from cumulative to per-step scores.
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        # Skip first EOS since it is always 0 and skews normalization.
        if self.opts.normalize_scores:
            eos_scores /= (step_nr + 1) ** self.opts.len_penalty

        # Holds the ids of finished searches.
        newly_finished: List[int] = []

        active_search_indices = (eos_beam_indices // self.beam_size).tolist()

        for beam_idx, search_idx in enumerate(active_search_indices):
            search_id, hypotheses = active_searches[search_idx]

            # We might have more than one beam finalized in one step that would
            # potentially exceed `beam_size` hypotheses.
            if len(hypotheses) == self.beam_size:
                continue

            hypotheses.append(
                Hypothesis(
                    seq=finalized_seqs[beam_idx],
                    score=eos_scores[beam_idx],
                    step_scores=finalized_scores[beam_idx],
                )
            )

            if len(hypotheses) == self.beam_size:
                # We have `beam_size` hypotheses for this particular search, so
                # we finish it now.
                newly_finished.append(search_idx)

                finished_searches[search_id] = hypotheses

        newly_finished.sort()

        # Remove finished searches from the active list.
        for idx in reversed(newly_finished):
            del active_searches[idx]

        return newly_finished


@dataclass
class SequenceGeneratorOutput:
    """Holds the output of a sequence generator."""

    results: List[List["Hypothesis"]]
    """The list of hypothesis generated per search, ordered by score."""

    device: Device
    """The device on which generated sequences reside."""

    collater: Optional[Collater] = None
    """The collater to use in :meth:`collate`."""

    def collate(
        self, hypo_idx: int = 0, skip_batch: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Collate the generated sequences at index ``hypo_idx`` in each search
        result into a single tensor.

        :param hypo_idx:
            The index of hypothesis to extract from each search result.
        :param skip_batch:
            If ``True``, if a search result has no hypothesis at index `hypo_idx`,
            it will be skipped instead of raising an error.

        :returns:
          - The collated sequences. *Shape:* :math:`(N,S)`, where :math:`N` is
            the number of search results and :math:`S` is the sequence length.
          - An array where each element represents the length of the sequence at
            the same index in the first returned value. *Shape:* :math:`(N)`,
            where :math:`N` is the number of search results.
        """
        if self.collater is None:
            raise RuntimeError("The output has no associated `Collater` instance.")

        if not self.results and not skip_batch:
            raise ValueError("The output must contain at least one search result.")

        seqs = []

        for search_idx, result in enumerate(self.results):
            if hypo_idx >= len(result):
                if not skip_batch:
                    raise ValueError(
                        f"Each search result must have at least {hypo_idx + 1} hypotheses, but search {search_idx} has only {len(result)}."
                    )

                continue

            seqs.append(result[hypo_idx].seq)

        if not seqs:
            # Return a zero-dimensional (not scalar!) tensor.
            return torch.empty((0,), device=self.device, dtype=torch.int64), None

        output = cast(SequenceData, self.collater(seqs))

        return output["seqs"], output["seq_lens"] if output["is_ragged"] else None


@dataclass
class Hypothesis:
    """Represents a hypothesis produced by a sequence generator."""

    seq: Tensor
    """The generated sequence."""

    score: Tensor
    """The score of the hypothesis."""

    step_scores: Tensor
    """The score of each individual sequence step."""
