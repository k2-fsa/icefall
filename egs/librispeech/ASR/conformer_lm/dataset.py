import torch
import torch.distributed as dist
import k2
import _k2
import sentencepiece as spm
from typing import Optional, List, Tuple



class LmDataset(torch.utils.data.Dataset):
    """
    Torch dataset for language modeling data.  This is a map-style dataset.
    The indices are integers.
    """
    def __init__(self, sentences: k2.RaggedInt,
                 words: k2.RaggedInt):
        super(LmDataset, self).__init__()
        self.sentences = sentences
        self.words = words


    def __len__(self):
        # Total size on axis 0, == num sentences
        return self.sentences.tot_size(0)

    def __getitem__(self, i: int):
        """
        Return the i'th sentence, as a list of ints (representing BPE pieces, without
        bos or eos symbols).
        """
        # It would be nicer if we could just return self.sentences[i].tolist(), but
        # for now that operator on k2.RaggedInt is not implemented.
        row_splits = self.sentences.row_splits(1)
        (begin, end) = row_splits[i:i+2].tolist()
        sentence = self.sentences.values()[begin:end]
        return k2.index(self.words, sentence).values().tolist()


def load_train_test_lm_dataset(archive_fn: str,
                               test_proportion: float = 0.025) -> Tuple[LmDataset, LmDataset]:
    """
    returns (train_lm_dataset, test_lm_dataset)
    """

    d = torch.load(archive_fn)
    words = d['words']  # a k2.RaggedInt with 2 axes, maps from word-ids to sequences of BPE pieces
    sentences = d['data']  # a k2.RaggedInt

    with torch.random.fork_rng(devices=[]):
        g = torch.manual_seed(0)
        num_sentences = sentences.tot_size(0)
        # probably the generator (g) argument to torch.randperm below is not necessary.
        sentence_perm = torch.randperm(num_sentences, generator=g, dtype=torch.int32)
        sentences = k2.index(sentences, sentence_perm)

    num_test_sentences = int(num_sentences * test_proportion)

    axis=0
    train_sents = _k2.ragged_int_arange(sentences, axis,
                                        num_test_sentences, num_sentences)
    test_sents = _k2.ragged_int_arange(sentences, axis, 0, num_test_sentences)

    return LmDataset(train_sents, words), LmDataset(test_sents, words)


def mask_and_pad(sentence: List[int],
                 seq_len: int,
                 bos_sym: int,
                 eos_sym: int,
                 blank_sym: int,
                 mask_proportion: float,
                 padding_proportion: float,
                 inv_mask_length: float,
                 unmasked_weight: float) -> Tuple[List[int], List[int], List[int], List[float]]:
    """
    This function contains part of the logic of collate_fn, broken out.  It is responsible
    for inserting masking and padding into the sequence `sentence`.  Most of the arguments
    are documented for `collate_fn` below.
    Other args:
        sentence: The original sentence to be masked and padded.
         seq_len: The desired length of the lists to be returned
         bos_sym, eos_sym, blank_sym, mask_proportion,
         padding_proportion, inv_mask_length, unmasked_weight: see their documentation
           as args to `collate_fn` below.


    Return: a tuple  (src, masked_src, tgt, weight, randomizable, attn_mask), all lists of length `seq_len`,
        where:
          `src` is: [bos] + [the sentence after inserting blanks in place of padding
                             after regions to be masked] + [eos] + [blank padding to seq_len].
          `src_masked` is as `src` but the masked regions have their values replaced with blank,
                i.e. they are actually masked.
          `tgt` is: [the original sentence, without masking] + [eos] + [blank] + [blank padding to seq_len]
          `weight` is the weight at the nnet output, which is: `unmasked_weight` for un-masked
                 positions, 1.0 for masked and padded positions, and 0.0 for positions that
                 correspond to blank-padding after the final [eos].
          `randomizable` is a bool that is True for positions where the symbol in
                 in `src_masked` is not bos or eos or blank.
          `attn_mask` is a bool that is False for positions in `src` and `src_masked` that
                 are between the initial [bos] and final [eos] inclusive; and True for
                 positions after the final [eos].
    """
    sent_len = len(sentence)
    assert sent_len + 3 <= seq_len

    for w in sentence:
        assert w not in [bos_sym, eos_sym, blank_sym]

    num_mask = int(torch.binomial(count=torch.tensor([sent_len * 1.0]),
                                  prob=torch.tensor([mask_proportion])).item())
    num_pad = int(torch.poisson(torch.tensor([sent_len * padding_proportion])).item())
    # Ensure the total length after bos, padding of masked sequences, and eos, is
    # no greater than seq_len
    num_pad -= max(0, sent_len + 2 + num_pad - seq_len)

    if num_mask + num_pad == 0:
        num_mask += 1

    # num_split_points is the number of times we split the (masked+padded)
    # region, so the total number of (masking+padding) subsequences will be
    # num_split_points + 1.  If num_mask positions are masked, then the
    # remaining number of words is `sent_len - num_mask`, and any two
    # masked regions must have at least one non-masked word between them,
    # so num_split_points == number of masked regions - 1, must be
    # no greater than `sent_len - num_mask`.  The formula about
    #   mask_proportion * inv_mask_length / (1.0 - mask_proportion)
    # is what's required (I think) so that inv_mask_length is the expected
    # length of masked regions.
    num_split_points = int(torch.binomial(count=torch.tensor([float(sent_len - num_mask)]),
                                          prob=torch.tensor([mask_proportion * inv_mask_length / (1.0 - mask_proportion)])).item())
    assert num_split_points <= sent_len - num_mask
    assert isinstance(num_split_points, int)

    def split_into_subseqs(length: int , num_subseqs: int) -> List[int]:
        """Splits a sequence of `length` items into `num_subseqs` possibly-empty
        subsequences.  The length distributions are geometric, not Poisson, i.e.
        we choose the split locations with uniform probability rather than
        randomly assigning each word to one subsequences.  This gives us more
        shorter/longer subsequences.
        Require num_subseqs > 0
        """
        boundaries = [0] + sorted(torch.randint(low=0, high=length + 1, size=(num_subseqs - 1,)).tolist()) + [length]
        return [ boundaries[i + 1] - boundaries[i] for i in range(num_subseqs) ]

    mask_lengths = split_into_subseqs(num_mask, num_split_points + 1)
    pad_lengths = split_into_subseqs(num_pad, num_split_points + 1)
    # mask_pad_lengths contains only the (mask, pad) length pairs for which mask + pad > 0.
    # From this point we only refer to the mask_pad_lengths.
    mask_pad_lengths = [ (mask, pad) for (mask, pad) in zip(mask_lengths, pad_lengths) if mask+pad > 0 ]
    num_subseqs = len(mask_pad_lengths)
    assert num_subseqs > 0

    # Now figure out how to distribute these subsequences throughout the actual
    # sentence.  The subsequences, if there are more than one, must not touch,
    # i.e. there must be an actual word in between each subsequence, where the
    # number of such "mandatory" words equals num_subseqs - 1.  We also have to
    # subtract `num_mask` words, since obviously the masked words cannot separate
    # the masked regions.
    reduced_len = sent_len - num_mask - (num_subseqs - 1)
    assert reduced_len >= 0
    # unmasked_lengths will be the lengths of the un-masked regions between the masked
    # regions.
    unmasked_lengths = split_into_subseqs(reduced_len, num_subseqs + 1)
    for i in range(1, num_subseqs):
        # Unmasked regions between masked regions must have length at least 1,
        # we add 1 to unmasked regions that are not initial/final.
        unmasked_lengths[i] = unmasked_lengths[i] + 1
    assert sum(unmasked_lengths) + sum(mask_lengths) == sent_len


    # src_positions will be: for each position in the masked+padded sentence,
    # the corresponding position in the source sentence `sentence`; or -1
    # if this was padding.
    src_positions = []
    # `masked` will be: for each position in the masked+padded sentence, True if
    # it was masked and False otherwise.  (Note: it is False for padding
    # locations, although this will not matter in the end).
    masked = []

    cur_pos = 0  # current position in source sentence
    for i in range(num_subseqs + 1):
        for j in range(unmasked_lengths[i]):
            src_positions.append(cur_pos)
            masked.append(False)
            cur_pos += 1
        if i < num_subseqs:
            (mask_len, pad_len) = mask_pad_lengths[i]
            for j in range(mask_len):
                src_positions.append(cur_pos)
                masked.append(True)
                cur_pos += 1
            for j in range(pad_len):
                src_positions.append(-1)
                masked.append(False)
    assert cur_pos == len(sentence)


    src = []
    src_masked = []
    tgt = []
    weight = []
    randomizable = []

    src.append(bos_sym)
    src_masked.append(bos_sym)
    randomizable.append(False)
    for i, src_pos in enumerate(src_positions):
        is_masked = masked[i]
        if src_pos >= 0:
            src_word = sentence[src_pos]
            src_masked.append(blank_sym if masked[i] else src_word)
            src.append(src_word)
            tgt.append(src_word)
            weight.append(1.0 if masked[i] else unmasked_weight)
            randomizable.append(not masked[i])
        else:
            # Padding inside a masked region
            src_masked.append(blank_sym)
            src.append(blank_sym)
            tgt.append(blank_sym)
            weight.append(1.0)
            randomizable.append(False)
    src.append(eos_sym)
    src_masked.append(eos_sym)
    tgt.append(eos_sym)
    weight.append(unmasked_weight)
    tgt.append(blank_sym)
    weight.append(0.0)
    randomizable.append(False)

    attn_mask = ([False] * len(src)) + ([True] * (seq_len - len(src)))

    for i in range(seq_len - len(src)):
        src.append(blank_sym)
        src_masked.append(blank_sym)
        tgt.append(blank_sym)
        weight.append(0.0)
        randomizable.append(False)

    return (src, src_masked, tgt, weight, randomizable, attn_mask)


# dataset.mask_and_pad(list(range(10, 20)), seq_len=16, bos_sym=1, eos_sym=2, blank_sym=0, mask_proportion=0.2, padding_proportion=0.2, inv_mask_length=0.33, unmasked_weight=0.444)

# dataset.collate_fn(sentences=[ list(range(10, 20)), list(range(30, 45))], bos_sym=1, eos_sym=2, blank_sym=0, mask_proportion=0.2, padding_proportion=0.2, randomize_proportion=0.05, inv_mask_length=0.33, unmasked_weight=0.444)

def collate_fn(sentences: List[List[int]],
               bos_sym: int,
               eos_sym: int,
               blank_sym: int,
               mask_proportion: float = 0.15,
               padding_proportion: float = 0.15,
               randomize_proportion: float = 0.05,
               inv_mask_length: float = 0.25,
               unmasked_weight: float = 0.25,
               debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
    """
    Caution, this is not the collate_fn we give directly to the dataloader,
    we give it a lambda: collate_fn=(lambda x: dataset.collate_fn(x, [other args]))
    This formats a list-of-lists-of-int into 5 Tensors, explained below.
    The key thing is that we mask out subsequences of random length within
    these sentences, and force the network to predict the masked-out
    subsequences (which have blanks appended to them to prevent the model
    from knowing the exact length of the sequences it has to predict).
    So it's like BERT but at the level of sequences rather than individual
    words.

    Args:
       bos_sym: the integer id of the beginning-of-sentence symbol, e.g. 2.
              Is allowed be the same as eos_sym (we are not necessarily
              saying it will work best that way).
       eos_sym: the integer id of the end-of-sentence symbol, e.g. 2.
       blank_sym:  the integer id of the blank symbol, e.g. 0 or 1.
       mask_proportion:  The proportion of words in each sentence that
              are masked, interpreted as (roughly) the probability of any given
              word being masked, although the masked locations will
              tend to be in contiguous sequences (they are not independent).
       padding_proportion: Like mask_proportion, but determines the
              number of extra, blank symbols that are inserted as padding
              at the end of masked regions (this ensures that the model
              cannot know exactly how many words need to be inserted in
              any given masked region.
       randomize_proportion:  The probability with which we replace
              words that were not masked with randomly chosen words.
              Like BERT, this is intended to force the model to predict
              something reasonable at non-masked positions, and to make
              this task harder than simply repeating the input.
       inv_mask_length: This number determines how many separate
              sub-sequences the (masked + padded) proportion of a sentence is split up
              into, interpreted as the inverse of the expected length of
              each *masked* region.
       unmasked_weight:  The weight to be applied to the log-likelihoods of
              un-masked positions in sentences (predicting un-masked
              positions is not completely trivial if randomize_proportion > 0).
              Will be reflected in the returned tgt_weights tensor.

    Returns a tuple (masked_src_symbols, src_symbols,
                     tgt_symbols, src_key_padding_mask,
                     tgt_weights),
         all with 2 axes and the same shape: (num_sent, seq_len).
         Their dtypes will be, respectively,
                   (torch.int64, torch.int64,
                    torch.int64, torch.bool,
                    torch.float)
         masked_src_symbols:  The sentences, with bos_symbol prepended and eos_symbol
                    appended, masked regions (including padding) replaced with blank,
                    and `randomize_proportion` non-masked symbols replaced with
                    symbols randomly taken from elsewhere in the sentences of this
                    minibatch.  Then padded to a fixed length with blank.
         src_symbols: Like masked_src_symbols, except with the masked symbols replaced
                    with the original symbols (but the padding that follows each
                    masked sub-sequence will still be blank)
         tgt_symbols: The original sentences, with eos_symbol appended, and then
                    padded with blank to the same length as masked_symbols and
                    src_symbols.
         src_key_padding_mask:  Masking tensor for masked_src_symbols and src_symbols, to
                    account for all the sentence lengths not being identical
                    (makes each sentence's processing independent of seq_len).
                    Tensor of Bool of shape (num_sent, seq_len), with True
                    for masked positions (these are the blanks that follow the
                    eos_symbol in masked_src_symbols), False for un-masked positions.
         tgt_weights:  Weights that will be applied to the log-probabilities at
                    the output of the network.  Will have 1.0 in positions
                    in `tgt_symbols` that were masked (including blank
                    padding at the end of masked regions), `unmasked_weight`
                    in other positions in the original sentences (including
                    terminating eos_symbol); and 0.0 in the remaining positions
                    corresponding to blank padding after the ends of
                    sentences.
    """
    assert blank_sym not in [bos_sym, eos_sym]
    max_sent_len = max([ len(s) for s in sentences])

    typical_mask_and_pad = int(max_sent_len * (mask_proportion + padding_proportion))

    # The following formula gives roughly 1 standard deviation above where we'd
    # expect the maximum sentence length to be with masking and padding.. we use
    # this as a hard upper limit, to prevent outliers from affecting the batch
    # size too much.  We use this as the size `seq_len`.
    # The "+ 4" is to ensure there is always room for the BOS, EOS and at least
    # two padding symbols.
    seq_len = max_sent_len + 4 + typical_mask_and_pad + int(typical_mask_and_pad ** 0.5)


    # srcs, srcs_masked, tgts and weights will be lists of the lists returned
    # from `mask_and_pad`, one per sentence.
    srcs = []
    srcs_masked = []
    tgts = []
    weights = []
    randomizables = []
    attn_masks = []
    for s in sentences:
        (src, src_masked, tgt,
         weight, randomizable,
         attn_mask) = mask_and_pad(s, seq_len, bos_sym, eos_sym,
                                   blank_sym, mask_proportion, padding_proportion,
                                   inv_mask_length, unmasked_weight)
        srcs.append(src)
        srcs_masked.append(src_masked)
        tgts.append(tgt)
        weights.append(weight)
        randomizables.append(randomizable)
        attn_masks.append(attn_mask)

    src_symbols = torch.tensor(srcs, dtype=torch.int64)
    masked_src_symbols = torch.tensor(srcs_masked, dtype=torch.int64)
    tgt_symbols = torch.tensor(tgts, dtype=torch.int64)
    src_key_padding_mask = torch.tensor(attn_masks, dtype=torch.bool)
    tgt_weights = torch.tensor(weights, dtype=torch.float)

    attn_mask_sum = torch.sum(torch.logical_not(src_key_padding_mask), dim=0).tolist()
    while attn_mask_sum[-1] == 0:  # Remove always-masked positions at the endof the lists.
        attn_mask_sum.pop()
    if len(attn_mask_sum) < seq_len:
        seq_len = len(attn_mask_sum)
        (src_symbols, masked_src_symbols,
         tgt_symbols, src_key_padding_mask, tgt_weights) = (src_symbols[:,:seq_len], masked_src_symbols[:,:seq_len],
                                                     tgt_symbols[:,:seq_len], src_key_padding_mask[:,:seq_len],
                                                     tgt_weights[:,:seq_len])

    if randomize_proportion > 0.0:
        randomizable_tensor = torch.tensor(randomizables, dtype=torch.bool)
        randomizable_indexes = torch.nonzero(randomizable_tensor)   # (num_randomizable, 2)
        num_randomizable = randomizable_indexes.shape[0]

        to_randomize_indexes = torch.nonzero(torch.rand(num_randomizable) < randomize_proportion, as_tuple=True)[0]
        num_to_randomize = to_randomize_indexes.numel()

        # older versions of torch don't have tensor_split, so fake a simplified version of it.
        # we'd be calling it as xxx.tensor_split(dim=1) if really in torc.
        def tensor_split(t):
            return (t[:,0], t[:,1])

        random_src_locations = torch.randperm(num_randomizable)[:num_to_randomize]

        random_symbols = src_symbols[tensor_split(randomizable_indexes[random_src_locations])]
        random_indexes_tuple= tensor_split(randomizable_indexes[to_randomize_indexes])
        src_symbols[random_indexes_tuple] = random_symbols
        masked_src_symbols[random_indexes_tuple] = random_symbols


    # I set this to true and tested with:
    # python3 -c 'import dataset; dataset.collate_fn(sentences=[ list(range(100, 200)), list(range(300, 450)), list(range(500,600))], bos_sym=1, eos_sym=2, blank_sym=0, mask_proportion=0.2, padding_proportion=0.2, randomize_proportion=0.05, inv_mask_length=0.33, unmasked_weight=0.444)'
    #.. and ran a few times to check the values printed looked about right, and that no assertions failed.
    if debug:
        check_collated_tensors(sentences, bos_sym, eos_sym, blank_sym,
                               unmasked_weight,
                               masked_src_symbols, src_symbols,
                               tgt_symbols, src_key_padding_mask, tgt_weights)
    return (masked_src_symbols, src_symbols,
            tgt_symbols, src_key_padding_mask, tgt_weights)



def check_collated_tensors(sentences: List[List[int]],
                           bos_sym: int,
                           eos_sym: int,
                           blank_sym: int,
                           unmasked_weight: float,
                           masked_src_symbols, src_symbols,
                           tgt_symbols, src_key_padding_mask,
                           tgt_weights):
    """
    This function checks the output of collate_fn, consider it test code.  Please see
    the documentation of collate_fn to understand the args.
    """
    for t in src_symbols, tgt_symbols, src_key_padding_mask, tgt_weights:
        assert t.shape == masked_src_symbols.shape

    tot_positions = src_symbols.numel()

    masked_src_symbols, src_symbols, tgt_symbols, src_key_padding_mask, tgt_weights = (
        masked_src_symbols.tolist(), src_symbols.tolist(), tgt_symbols.tolist(),
        src_key_padding_mask.tolist(), tgt_weights.tolist())
    assert len(sentences) == len(masked_src_symbols)

    tot_masked_positions = 0
    tot_padded_positions = 0
    tot_unmasked_positions = 0  # all un-masked, non-blank postions, including eos
    tot_randomized_positions = 0
    num_masked_subseqs = 0
    tot_symbols = 0  # original symbols in sentences, no bos/eos

    assert unmasked_weight > 0.001  # or this test code won't work..

    for i in range(len(sentences)):
        reconstructed_sent = list(filter(lambda x: x not in [bos_sym,eos_sym,blank_sym], tgt_symbols[i]))
        if sentences[i] != reconstructed_sent:
            print(f"Error: sentence {i}={sentences[i]} differs from {reconstructed_sent}")
        (masked_src, src, tgt, src_mask, weights) = (masked_src_symbols[i], src_symbols[i],
                                                     tgt_symbols[i], src_key_padding_mask[i], tgt_weights[i])

        assert src[0] == masked_src[0] == bos_sym
        for j in range(len(masked_src)):
            assert masked_src[j] == blank_sym or masked_src[j] == src[j]

            if src[j] not in [bos_sym, eos_sym, blank_sym]:
                tot_symbols += 1

            if j > 0:
                assert (src[j] == eos_sym) == (masked_src[j] == eos_sym) == (tgt[j-1] == eos_sym)
                if masked_src[j] == blank_sym:  # masked or padding of masked subseq, or post-eos padding..
                    assert src[j] == tgt[j - 1]  # masked symbols are not randomized.
                    assert weights[j - 1] in [0.0, 1.0]  # 0.0 for final blank padding
                    if weights[j - 1] == 1.0:  # Not final blank padding...
                        if tgt[j - 1] == blank_sym:
                            tot_padded_positions += 1
                        else:
                            tot_masked_positions += 1
                        if masked_src[j + 1] != blank_sym:
                            num_masked_subseqs += 1
                else:
                    assert weights[j - 1] == 0 or abs(weights[j-1] - unmasked_weight) < 0.001
                    if abs(weights[j - 1]-unmasked_weight) < 0.001:
                        tot_unmasked_positions += 1
                        if tgt[j - 1] != src[j]:
                            tot_randomized_positions += 1

            if src_mask[j]:  # if masked..
                assert src[j] == blank_sym

    assert tot_symbols == sum(len(x) for x in sentences)

    assert tot_unmasked_positions + tot_masked_positions == tot_symbols + len(sentences)

    print(f"{tot_unmasked_positions} + {tot_masked_positions} == {tot_symbols} + {len(sentences)}")
    print(f"tot_symbols / tot_positions = {tot_symbols/tot_positions} (rest is bos,eos,padding)")

    print(f"Masking/tot_symbols = {tot_masked_positions/tot_symbols}, Padding/tot_symbols = {tot_padded_positions/tot_symbols}")
    print(f"Randomization/tot_non_masked_symbols = {tot_randomized_positions/(tot_symbols-tot_masked_positions)}")
    print(f"Mean masking length = {tot_masked_positions/num_masked_subseqs}, Mean padding length = {tot_padded_positions/num_masked_subseqs}")



# This shows some useful code about the BPE encoding.
#    import sentencepiece as spm
#    sp = spm.SentencePieceProcessor()
#    sp.load(bpe_model_fn)  # bpe.model
#    sp.GetPieceSize(..)
#    sp.Decode(...)
#    sp.Encode(...)


# import dataset
# import torch
# train,test = dataset.load_train_test_lm_dataset('../data/lm_training_5000/lm_data.pt')


# train_dl = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, collate_fn=(lambda x: train.collate_fn(x)))
# x = iter(train_dl)
# str(next(x))
#  '[ [ 10 38 651 593 3 1343 31 780 6 4172 112 788 1696 24 289 24 3 403 6 4493 162 92 71 328 417 217 338 14 5 3 1876 154 21 23 2237 43 3 1535 92 71 2816 7 1031 31 2318 92 2528 4806 14 206 3 954 1373 6 525 4 631 447 2639 ] [ 1014 336 171 209 795 10 16 90 27 787 139 53 45 2817 ] [ 11 980 51 22 1748 14 91 105 363 428 6 8 2887 3305 2525 2297 70 3 4651 6 27 282 335 426 134 292 5 193 3 539 2250 584 127 ] [ 9 3 1858 4 18 2257 4 6 41 748 10 304 7 229 83 2793 4 9 981 7 1484 33 3 103 7 539 5 477 3195 18 64 39 82 1034 6 3 4128 ] [ 17 147 22 7 708 60 133 174 105 4111 4 6 3 1384 65 50 1051 9 2953 6 3 461 180 1142 23 5 36 888 8 131 173 390 78 23 266 2822 715 46 182 65 22 1739 33 3 700 1450 14 233 4 ] [ 80 10 16 67 279 7 1827 264 96 3 187 2851 2108 ] [ 1473 48 106 227 9 160 2011 4 674 ] [ 3 954 762 29 85 228 33 8 940 40 4952 36 486 390 595 3 81 225 6 1440 125 346 134 296 126 419 1017 3824 4 8 179 184 11 33 580 1861 ] [ 30 22 245 15 117 8 2892 28 1204 145 7 3 236 3417 6 3 3839 5 3106 155 198 30 228 2555 46 15 32 41 747 72 9 25 977 ] [ 222 466 6 3157 ] ]'
#
# or:
# import k2
# k2.ragged.to_list(next(x))
# [shows something similar].
#
# You'd really do something like:
#   for epoch in range(max_epochs):
#      for minibatch in train_dl:


# .. How to process data?  Suppose we have a sentence like [259, 278, 45, 11, 303, 1319, 34, 15, 396, 3435, 7, 44].
#
#  First: we randomly choose one or more starting positins for a masked segment.
#     Each sentence must have at least one masked segment (or there is no contribution to the loss function).
#     We choose to have:
#          num_masked_segments = max(1, len(sent) // 15)
#
#     The length of the masked segment (this is the target for prediction), we set to the geometric
#     distribution with the probability of success set to 3:
#
#      g = torch.distributions.geometric.Geometric(probs=0.3)   # <-- expected value is 3.333
#   Example of sampling:
#      g.sample(sample_shape=torch.Size([10]))
#
#   We now we randomly compute the location of the masked segments (length computed above) as follows:
#   First, the masked segments must be separated by at least one non-masked word (else they would be
#   a single segment).  So for n masked segments, there are n-1 words required for minimal separation.
#   If tot-length-of-segments + n-1 is greater than the sentence length, we just have the entire
#   sentence be masked.  Otherwise, we randomly divide the remaining number of words between the n+1
#   positions where they can appear (e.g. for 2 segments, this would be at the start, between the 2 segments,
#   and at the end).  This is the multinomial distribution, but we can more easily compute this
#   directly using rand() and cutoffs, rather than creating a torch.distributions.Multinomial().
#

#  Next we need to compute a random amount of blank padding (>= 0) for each of the masked regions;
#  this is done so the model never knows the exact length of the masked region.  We can just use the
#  same distribution as for the length of the masked regions, i.e. geometric with success-prob=0.3
#  (expected padding length is 3).
#
#  At this point we know where the masked regions are and how much padding they have.  We can format
#  the result as three lists, of the same length:
#
#      sent:       contains the words in the sentence with, in masked
#                  positions, the original (target) words, then with
#                  blank in the blank-padding after masked positions.
#
#   sent_augmented:  `sent` with, at a small defined percentage of positions
#                  that were *not* masked, the real token replaced with a
#                  token randomly chosen from the tokens in the minibatch.
#                  (like BERT, we use this type of augmentation, so the model
#                  has to predict the original token).
#
#    masked_sent_augmented: List[int], contains the words in `sent_augmented`, except
#                  with masked  positions and the blank padding after the masked regions
#                  both replaced with blank.
#
#
#
#  The way these will be processed is as follows:
#
#    masked_sent_in = [bos] + masked_sent_augmented + [eos] <-- so we know the sentence ended, distinguish it from truncated ones.
#           sent_in = [bos] + sent_augmented + [eos]
#
#     sent_out = sent + [eos] + [eos]     #<--- the predicted targets at each point, although
#                                         #     we only really care about this in masked regions.
#                                         #  The extra eos is so that the length is the same as
#                                         #  masked_sent_in and sent_in.
#
#    out_scale = (masked_sent==blk ? 1.0 : non_masked_scale)  # e.g. non_masked_scale = 1.0 is fine,
#                                                             # this is a choice; we can perhaps
#                                                             # report these 2 parts of the loss
#                                                             # separately though.
#                                                             # <-- can also set the last element
#                                                             #   of out_scale to a smaller number, since
#                                                             # it's a repeated eos.
#
#
# OK, how do we combine these into a minibatch?  Firstly, we truncate sentences to a maximum
# length, e.g. 128, if `masked_sent_in`/`sent_in` have length longer than that.  We choose randomly
# in each case to truncate the beginning or end, truncating both masked_sent_in/sent_in and sent_out
# from the same side.  Caution: this means that these sentences may lack bos and/or eos symbols.
#
# Next, we combine shorter utterances by appending them  ( all of:  masked_sent_in, sent_in, out_scale)
# as long as doing so would keep the total length under 128.  We then pad (masked_sent_in, sent_in, sent_out, out_scale)
# with:  (<blk>,<blk>,<eos>, 0) up to the maximum length of any sentence in the minibatch <- or could use
#
#
#
#
#
#
#
#                                  # i.e. ones where masked_sent is blank and zeros elsewhere;
#                                 #  this pertains to positions in `sent_out`.
#
#
#
#
#
#
#
#
#
#
# torch.distributions.gamma.Gamma(concentration=1.0, rate=1.0/5)




class LmBatchSampler(torch.utils.data.Sampler):
    """
    A sampler that returns a batch of integer indexes as a list, intended for use
    with class LmDataset.  The sentences returned in each batch will all be about
    the same size, and the batch size is specified as a number of words (we also
    provide an option that allows you to limit the max memory consumed by transformers)

    Has support for distributed operation.
    """
    def __init__(self, dataset: LmDataset,
                 symbols_per_batch: int,
                 quadratic_constant: float = 0.005,
                 world_size: Optional[int] = None,
                 rank: int = None,
                 seed: int = 0):
        """
        Constructor documentation:
           dataset:  the LmDataset object that we are sampling from.  This
              class does not retain a reference to the LmDataset.
       symbols_per_batch:  The number of BPE symbols desired in each minibatch
       quadratic_constant:  After the sentence length gets more than about
                  1.0/quadratic_constant, the batch size will start decreasing
                  as 1/(sentence-length^2).  This is a mechanism to
                  avoid excessive memory consumption in transformers, when
                  sentence length gets long.
       world_size:  The world size for distributed operation; if None,
                  will be worked out from torch.distributed.
       rank:  The rank of this sampler/process for distributed operation; if None,
                  will be worked out from torch.distributed.
       seed:  The random seed
        """
        self.seed = seed
        self.symbols_per_batch = symbols_per_batch
        self.quadratic_constant = quadratic_constant
        self._maybe_init_distributed(world_size=world_size, rank=rank)

        # a configuration constant we don't expose.
        self.multiplicative_random_length = 0.05

        # "indexes" is the subset of indexes into LmDataset that this
        # sampler is reponsible for (all of them, in the non-distributed case).
        data_indexes = torch.arange(self.rank, len(dataset), self.world_size, dtype=torch.int32)  # dtype=torch.int32

        word_row_splits = dataset.words.row_splits(1)  # dtype=torch.int32
        word_lengths = word_row_splits[1:] - word_row_splits[:-1]  # dtype=torch.int32

        # the sentences this sampler is responsible for, as sequences of words.
        # It's a ragged tensor of int32
        sentences = k2.index(dataset.sentences, data_indexes)

        # sentence_lengths is a k2.RaggedInt like `sentences`, but with the words replaced
        # with their respective lengths, in BPE pieces.
        sentence_lengths = k2.index(word_lengths, sentences)
        del sentences # save memory
        assert isinstance(sentence_lengths, k2.RaggedInt)

        # convert to float so sum_per_sublist() will work (TODO: sum_per_sublist() will eventually
        # support int32.)
        sentence_lengths = k2.RaggedFloat(sentence_lengths.shape(),
                                           sentence_lengths.values().to(torch.float32))
        assert isinstance(sentence_lengths, k2.RaggedFloat)

        # Convert into a simple tensor of float by adding lengths of words.
        sentence_lengths = k2.ragged.sum_per_sublist(sentence_lengths)

        assert isinstance(sentence_lengths, torch.Tensor)
        assert sentence_lengths.dtype == torch.float32

        # self.sentence_lengths is a Tensor with dtype=torch.float32.  It
        # contains the lengths, in BPE tokens, of the sentences that this
        # sampler is responsible for, whose real indexes are in
        # `data_indexes` above (this is not stored, as we know the formula).
        self.sentence_lengths = sentence_lengths

        self.set_epoch(0)  # this is responsible for setting self.sorted_data_indexes


    def _maybe_init_distributed(self, world_size: Optional[int], rank: Optional[int]):
        if world_size is not None:
            assert world_size >= 1
        if rank is not None:
            assert rank >= 0
        if not dist.is_available() or not dist.is_initialized():
            self.world_size = 1 if world_size is None else world_size
            self.rank = 0 if rank is None else rank
            return
        self.world_size = dist.get_world_size() if world_size is None else world_size
        self.rank = dist.get_rank() if rank is None else rank
        assert self.rank < self.world_size

    def set_epoch(self, epoch: int):
        """
        Must be called at the beginning of each epoch, before initializing the DataLoader,
        to re-shuffle the data.   If this is not done, this sampler will give you the same batches
        each time it is called.
        """
        g = torch.manual_seed(self.rank + self.seed + epoch)

        sentence_lengths = (self.sentence_lengths *
                            (1.0 + torch.rand(*self.sentence_lengths.shape, generator=g) * self.multiplicative_random_length))

        # This mechanism regulates the batch size so that we don't get OOM in transformers
        # when the sentences are long.
        sentence_lengths = sentence_lengths + (sentence_lengths ** 2) * self.quadratic_constant

        values, indices = torch.sort(sentence_lengths) # values,indices dtypes: torch.float,torch.int64

        # map to the original indexes into the dataset (the original sentence
        # indexes), see torch.arange expression in the constructor.  save as
        # int32 just to save a little memory.  self.indices are indexes into the
        # LmDataset, just including the subset of indices that this sampler is
        # responsible for (in terms of rank and world_size), and sorted by
        # length with a small amount of randomization specific to the epoch.
        self.indices = ((indices * self.world_size) + self.rank).to(dtype=torch.int32)

        # now `batch_ids` will be: [0, 0, 0, 0, .., 0, 1, 1, 1, ... 1, 2, ... ],
        # saying which batch each element of values/indices belongs to.
        batch_ids = (torch.cumsum(values, dim=0) * (1.0 / self.symbols_per_batch)).to(dtype=torch.int32)

        batch_boundaries = torch.nonzero(batch_ids[1:] - batch_ids[:-1], as_tuple=True)[0]
        batch_boundaries.add_(1)
        self.batch_boundaries = torch.cat((torch.zeros(1, dtype=torch.int32), batch_boundaries), dim=0)

        num_batches = self.batch_boundaries.numel() - 1

        # self.batch_indices is a permutation of [0, 1, ... num_batches -
        # 1]; it determines the order in which we access the batches.  It's
        # necessary to randomize the order of these, to avoid returning batches
        # from shortest to longest sentences.
        self.batch_indices = torch.randperm(num_batches, generator=g, dtype=torch.int32).tolist()


    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        """
        Iterator that yields lists of indices (i.e., integer indices into the LmDataset)
        """
        for batch_idx in self.batch_indices:
            batch_start = self.batch_boundaries[batch_idx].item()
            batch_end = self.batch_boundaries[batch_idx + 1].item()
            yield self.indices[batch_start:batch_end].tolist()






# train,test = dataset.load_train_test_lm_dataset('../data/lm_training_5000/lm_data.pt')
# sampler = dataset.LmBatchSampler(test, symbols_per_batch=1000, world_size=2, rank=0)
# a = iter(sampler)
# print(str(next(a)))

# collate_fn=(lambda x:dataset.collate_fn(x, bos_sym=1, eos_sym=1, blank_sym=0, debug=True))
# train_dl = torch.utils.data.DataLoader(test, batch_sampler=sampler, collate_fn=collate_fn)
# x = iter(train_dl)
# print(str(next(x)))
