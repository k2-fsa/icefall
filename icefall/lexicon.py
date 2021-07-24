import logging
import re
from pathlib import Path
from typing import List

import k2
import torch


class Lexicon(object):
    """Phone based lexicon.

    TODO: Add BpeLexicon for BPE models.
    """

    def __init__(
        self, lang_dir: Path, disambig_pattern: str = re.compile(r"^#\d+$")
    ):
        """
        Args:
          lang_dir:
            Path to the lang director. It is expected to contain the following
            files:
                - phones.txt
                - words.txt
                - L.pt
            The above files are produced by the script `prepare.sh`. You
            should have run that before running the training code.
          disambig_pattern:
            It contains the pattern for disambiguation symbols.
        """
        lang_dir = Path(lang_dir)
        self.phones = k2.SymbolTable.from_file(lang_dir / "phones.txt")
        self.words = k2.SymbolTable.from_file(lang_dir / "words.txt")

        if (lang_dir / "Linv.pt").exists():
            logging.info("Loading pre-compiled Linv.pt")
            L_inv = k2.Fsa.from_dict(torch.load(lang_dir / "Linv.pt"))
        else:
            logging.info("Converting L.pt to Linv.pt")
            L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt"))
            L_inv = k2.arc_sort(L.invert())
            torch.save(L_inv.as_dict(), lang_dir / "Linv.pt")

        # We save L_inv instead of L because it will be used to intersect with
        # transcript, both of whose labels are word IDs.
        self.L_inv = L_inv
        self.disambig_pattern = disambig_pattern

    @property
    def tokens(self) -> List[int]:
        """Return a list of phone IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a phone ID so it is excluded from the return value.
        """
        symbols = self.phones.symbols
        ans = []
        for s in symbols:
            if not self.disambig_pattern.match(s):
                ans.append(self.phones[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans
