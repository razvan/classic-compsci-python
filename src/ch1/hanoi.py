#!/usr/bin/env python3

from typing import TypeVar, Generic, List

T = TypeVar("T")


class Stack(Generic[T]):
    def __init__(self) -> None:
        self._c: List[T] = []

    def push(self, o: T) -> None:
        self._c.append(o)

    def pop(self) -> T:
        return self._c.pop()

    def __repr__(self) -> str:
        return repr(self._c)


class Hanoi:
    def __init__(self, num_discs: int = 3) -> None:
        self.num_discs: int = num_discs
        self.tower_a: Stack[int] = Stack()
        self.tower_b: Stack[int] = Stack()
        self.tower_c: Stack[int] = Stack()
        for i in range(1, num_discs + 1):
            self.tower_a.push(i)

    def __repr__(self) -> str:
        """
        >>> Hanoi()
        Hanoi(num_discs: 3, tower_a: [1, 2, 3], tower_b: [], tower_c: [])
        """
        return "Hanoi(num_discs: {}, tower_a: {}, tower_b: {}, tower_c: {})".format(
            self.num_discs, repr(self.tower_a), repr(self.tower_b), repr(self.tower_c)
        )

    def _solve_rec(
        self, begin: Stack[int], end: Stack[int], temp: Stack[int], n: int
    ) -> None:
        if n == 1:
            end.push(begin.pop())
        else:
            self._solve_rec(begin, temp, end, n - 1)
            self._solve_rec(begin, end, temp, 1)
            self._solve_rec(temp, end, begin, n - 1)

    def run(self) -> "Hanoi":
        """
        >>> Hanoi().run()
        Hanoi(num_discs: 3, tower_a: [], tower_b: [], tower_c: [1, 2, 3])
        """
        self._solve_rec(self.tower_a, self.tower_c, self.tower_b, self.num_discs)
        return self
