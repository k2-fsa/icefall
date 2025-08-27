# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import kaldifst


# Note the name contains `standard`; it means there will be non-standard
# topologies.
def build_standard_ctc_topo(max_token_id: int) -> kaldifst.StdVectorFst:
    """Build a standard CTC topology.

    Args:
      Maximum valid token ID. We assume token IDs are contiguous
      and starts from 0. In other words, the vocabulary size is
      ``max_token_id + 1``. We assume the ID of the blank symbol is 0.
    """
    # Token ID starts from 0 and there are as many states as the
    # number of tokens.
    #
    # Note that epsilon is not a token and the token with ID 0 in tokens.txt
    # is not an epsilon. It means input label 0 of the resulting FST does
    # not represent an epsilon.
    #
    # You can use the function `add_one()` to modify the input/output labels
    # of the resulting FST

    num_states = max_token_id + 1

    # Step 1: Create as many states as the number of tokens.
    # Each state is a final state
    fst = kaldifst.StdVectorFst()
    for i in range(num_states):
        s = fst.add_state()
        fst.set_final(state=s, weight=0)

    # Step 2: Set state 0 as the start state.
    # We assume the ID of the blank symbol is 0.
    fst.start = 0

    # Step 3: Build a fully connected graph.
    for i in range(num_states):
        for k in range(num_states):
            fst.add_arc(
                state=i,
                arc=kaldifst.StdArc(
                    ilabel=k,
                    olabel=k if i != k else 0,  # if i==k, it is a self loop
                    weight=0,
                    nextstate=k,
                ),
            )
    # Please see ./test_ctc_topo.py if you want to know what the resulting
    # FST looks like

    return fst


def add_one(
    fst: kaldifst.StdVectorFst,
    treat_ilabel_zero_specially: bool,
    update_olabel: bool,
) -> None:
    """Modify the input and output labels of the given FST in-place.

    Args:
      fst:
        The FST to be modified. It is changed in-place.
      treat_ilabel_zero_specially:
        If True, then every non-zero input label is increased by one and the
        zero input label is not changed.
        If False, then every input label is increased by one.
      update_olabel:
        If False, the output label is not changed.
        If True, then every non-zero output label is increased by one.
        In either case, output label with 0 is not changed.
    """
    for state in kaldifst.StateIterator(fst):
        for arc in kaldifst.ArcIterator(fst, state):
            # If treat_ilabel_zero_specially is False, we always change it
            # Otherwise, we only change non-zero input labels
            if treat_ilabel_zero_specially is False or arc.ilabel != 0:
                arc.ilabel += 1

            if update_olabel and arc.olabel != 0:
                arc.olabel += 1

    if fst.input_symbols is not None:
        input_symbols = kaldifst.SymbolTable()
        input_symbols.add_symbol(symbol="<eps>", key=0)

        for i in range(0, fst.input_symbols.num_symbols()):
            s = fst.input_symbols.find(i)
            input_symbols.add_symbol(symbol=s, key=i + 1)

        fst.input_symbols = input_symbols

    if update_olabel and fst.output_symbols is not None:
        output_symbols = kaldifst.SymbolTable()
        output_symbols.add_symbol(symbol="<eps>", key=0)

        for i in range(0, fst.output_symbols.num_symbols()):
            s = fst.output_symbols.find(i)
            output_symbols.add_symbol(symbol=s, key=i + 1)

        fst.output_symbols = output_symbols


def add_disambig_self_loops(fst: kaldifst.StdVectorFst, start: int, end: int):
    """Add self-loops to each state.

    For each disambig symbol, we add a self-loop with input label disambig_id
    and output label diambig_id of that disambig symbol.

    Args:
      fst:
        It is changed in-place.
      start:
        The ID of #0
      end:
        The ID of the last disambig symbol. For instance if there are 3
        disambig symbols ``#0``, ``#1``, and ``#2``, then ``end`` is the ID
        of ``#2``.
    """
    for state in kaldifst.StateIterator(fst):
        for i in range(start, end + 1):
            fst.add_arc(
                state=state,
                arc=kaldifst.StdArc(
                    ilabel=i,
                    olabel=i,
                    weight=0,
                    nextstate=state,
                ),
            )

    if fst.output_symbols:
        for i in range(start, end + 1):
            fst.output_symbols.add_symbol(symbol=f"#{i-start}", key=i)
