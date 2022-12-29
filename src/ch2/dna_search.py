#!/usr/bin/env python3

from enum import IntEnum
from typing import Tuple, List, Generator

Nucleotide: IntEnum = IntEnum("Nucleotide", ("A", "C", "G", "T"))
Codon = Tuple[Nucleotide, Nucleotide, Nucleotide]
Gene = Generator[Codon, None, None]


def to_gene(s: str) -> Gene:
    """
    >>> next(to_gene("ACGTGGCTCTCTAACGTACGTACGTACGGGGTTTATATATACCCTAGGACTCCCTTT"))
    (<Nucleotide.A: 1>, <Nucleotide.C: 2>, <Nucleotide.G: 3>)
    """
    for i in range(0, len(s), 3):
        if i + 3 < len(s):
            yield (Nucleotide[s[i]], Nucleotide[s[i + 1]], Nucleotide[s[i + 2]])


def linear_contains(what: Codon, where: Gene) -> bool:
    """
    >>> gene = to_gene("ACGTGGCTCTCTAACGTACGTACGTACGGGGTTTATATATACCCTAGGACTCCCTTT")
    >>> acg: Codon = (Nucleotide.A, Nucleotide.C, Nucleotide.G)
    >>> linear_contains(acg, gene)
    True
    >>> gat: Codon = (Nucleotide.G, Nucleotide.A, Nucleotide.T)
    >>> linear_contains(gat, gene)
    False
    """
    for c in where:
        if what == c:
            return True
    return False
