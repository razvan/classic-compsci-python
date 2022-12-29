#!/usr/bin/env python3

from enum import StrEnum
from typing import NamedTuple, List, Optional
from dataclasses import dataclass, field
from itertools import product
from random import uniform


class Cell(StrEnum):
    EMPTY = " "
    BLOCKED = "X"
    START = "S"
    GOAL = "G"
    PATH = "*"


class MazeLocation(NamedTuple):
    row: int
    col: int


@dataclass
class Maze:
    sparseness: float = field(default=0.2)
    rows: int = field(default=10)
    cols: int = field(default=10)
    start: MazeLocation = field(default=MazeLocation(0, 0))
    goal: MazeLocation = field(default=MazeLocation(9, 9))
    _grid: List[List[Cell]] = field(repr=False, init=False)

    def __post_init__(self) -> None:
        self._fill_maze()

    def _fill_maze(self) -> None:
        self._grid = [[Cell.EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        for (i, j) in product(range(self.rows), range(self.cols)):
            if uniform(0.0, 1.0) < self.sparseness:
                self._grid[i][j] = Cell.BLOCKED
            else:
                self._grid[i][j] = Cell.EMPTY
        self._grid[self.start.row][self.start.col] = Cell.START
        self._grid[self.goal.row][self.goal.col] = Cell.GOAL

    def __str__(self) -> str:
        output = ""
        for row in self._grid:
            output += "".join([c for c in row]) + "\n"
        return output

    def surrounding_locations(
            self, cur: MazeLocation, dist: int = 1, exclude: Optional[Cell] = Cell.BLOCKED
    ) -> List[MazeLocation]:
        """
        Return all maze locations surrounding `cur`.

        Parameters
        ----------
        cur : MazeLocation
            The location for which to get it's surroudings.
        dist : int
            Size of the surrouding rectangle (default 1)
        exclude : Optional[Cell], optional
            Exclude locations that occupied by the provided Cell (default is Cell.BLOCKED)

        Returns
        ------
        List[MazeLocation]
            All locations within distance `dist` of `cur` that are not `exclude`.
        """
        # rectangle centered in "cur"
        result = [
            (x, y)
            for x in range(cur.row - dist, cur.row + dist + 1)
            for y in range(cur.col - dist, cur.col + dist + 1)
        ]
        # remove out of bounds locations
        result = filter(
            lambda t: 0 <= t[0] < self.rows and 0 <= t[1] < self.cols, result
        )
        # remove "cur" it's self
        result = filter(lambda t: t != cur, result)
        # remove any provided "exclude"
        if exclude != None:
            result = filter(lambda t: self._grid[t[0]][t[1]] != exclude, result)
        return [MazeLocation(x, y) for (x, y) in result]


if __name__ == "__main__":
    m = Maze(cols=11)
    print(m)
    print(list(m.surrounding_locations(MazeLocation(1, 1))))
