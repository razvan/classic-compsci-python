#!/usr/bin/env python3

from enum import IntEnum
from typing import TypeVar, Type


class Error(BaseException):
    pass


T = TypeVar('T', bound='Nucleotide')


class Nucleotide(IntEnum):
    A: int = 0b00
    C: int = 0b01
    G: int = 0b10
    T: int = 0b11

    @classmethod
    def try_from_str(cls: Type[T], s: str) -> T:
        for m in Nucleotide:
            if s == m.name:
                return m
        raise Error(f"invalid nucleotide {s}")

    @classmethod
    def try_from_bits(cls: Type[T], b: int) -> T:
        for m in Nucleotide:
            if b == m.value:
                return m
        raise Error(f"invalid nucleotide {b}")


class CompressedGene:
    """
    Compress sequences of nucleotides by using only two bits per value.

    >>> from sys import getsizeof
    >>> original: str = "ACGT" * 1000
    >>> getsizeof(original)
    4049
    >>> getsizeof(CompressedGene(original).bstr)
    1092
    """
    def __init__(self, gene: str) -> None:
       self._try_compress(gene)

    def _try_compress(self, gene: str) -> None:
        self.bstr: int = 1
        for ns in gene:
            self.bstr <<= 2
            self.bstr |= Nucleotide.try_from_str(ns).value

    def decompress(self) -> str:
        gene: str = ""
        for i in range(0, self.bstr.bit_length()-1, 2):
            bits: int = self.bstr >> i & 0b11
            gene += Nucleotide.try_from_bits(bits).name
        return gene[::-1]

    def __str__(self) -> str:
        return self.decompress()
