"""
Missionaries and cannibals

"""
from __future__ import annotations
from attrs import define, field, Factory
from typing import List, Optional, Iterator
from .maze import bfs, to_path

MAX_NUM: int = 3


@define(frozen=True)
class MCState:
    missionaries: int
    cannibals: int
    boat: bool = True
    _wm: int = field(init=False, default=Factory(
        lambda self: self.missionaries, takes_self=True))
    _wc: int = field(init=False, default=Factory(
        lambda self: self.cannibals, takes_self=True))
    _em: int = field(init=False, default=Factory(
        lambda self: MAX_NUM-self.missionaries, takes_self=True))
    _ec: int = field(init=False, default=Factory(
        lambda self: MAX_NUM-self.cannibals, takes_self=True))

    @property
    def is_legal(self) -> bool:
        if self._wm > 0 and self._wm < self._wc:
            return False
        if self._em > 0 and self._em < self._ec:
            return False
        return True

    def goal_test(self) -> bool:
        return self.is_legal and self._em == MAX_NUM and self._ec == MAX_NUM

    def successors(self) -> List[MCState]:
        result: List[MCState] = []
        if self.boat:  # west
            if self._wm > 1:
                result.append(MCState(self._wm - 2, self._wc, not self.boat))
            if self._wm > 0:
                result.append(MCState(self._wm - 1, self._wc, not self.boat))
            if self._wc > 1:
                result.append(MCState(self._wm, self._wc - 2, not self.boat))
            if self._wc > 0:
                result.append(MCState(self._wm, self._wc - 1, not self.boat))
            if (self._wm > 0) and (self._wc > 0):
                result.append(
                    MCState(self._wm - 1, self._wc - 1, not self.boat))
        else:  # east
            if self._em > 1:
                result.append(MCState(self._wm + 2, self._wc, not self.boat))
            if self._em > 0:
                result.append(MCState(self._wm + 1, self._wc, not self.boat))
            if self._ec > 1:
                result.append(MCState(self._wm, self._wc + 2, not self.boat))
            if self._ec > 0:
                result.append(MCState(self._wm, self._wc + 1, not self.boat))
            if (self._em > 0) and (self._ec > 0):
                result.append(
                    MCState(self._wm + 1, self._wc + 1, not self.boat))
        return [s for s in result if s.is_legal]


def display_solution(path: Iterator[MCState]) -> None:
    old_state = next(path)
    print(
        f"West is starting with [{old_state.missionaries}] missionaries and [{old_state.cannibals}] cannibals")
    for s in path:
        if s.boat:
            print("E -> W moved [{}] missionaries and [{}] cannibals (W={}m+{}c, E={}m+{}c, Boat=West)".format(
                old_state._em - s._em, old_state._ec - s._ec, s._wm, s._wc, s._em, s._ec))
        else:
            print("W -> E moved [{}] missionaries and [{}] cannibals (W={}m+{}c, E={}m+{}c, Boat=East)".format(
                old_state._wm - s._wm, old_state._wc - s._wc, s._wm, s._wc, s._em, s._ec))
        old_state = s


def main():
    start = MCState(MAX_NUM, MAX_NUM, True)
    solution = bfs(start, MCState.goal_test, MCState.successors)
    if solution is None:
        print("No solution found!")
    else:
        path = to_path(solution)
        display_solution(path)


if __name__ == '__main__':
    main()
