#!/usr/bin/env python3

from functools import partial
from collections import deque
from enum import StrEnum
from typing import Deque, Iterator, NamedTuple, List, Optional, Generic, TypeVar, Self, Callable, Set
from dataclasses import dataclass, field
from itertools import product
from random import uniform, seed
from copy import deepcopy


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

    def successors(
            self, cur: MazeLocation, dist: int = 1, exclude: Optional[Cell] = Cell.BLOCKED
    ) -> List[MazeLocation]:
        """
        Return all valid maze locations surrounding `cur` where a move can occur.

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
        if dist < 1:
            raise ValueError("dist cannot be smaller than 1")

        result = []
        for offset in range(1, dist+1):
            result.extend([t for t in [
                MazeLocation(cur.row - offset, cur.col),  # north
                MazeLocation(cur.row, cur.col + offset),  # east
                MazeLocation(cur.row + offset, cur.col),  # south
                MazeLocation(cur.row, cur.col - offset),  # west
            ] if self._valid_location(cur, t, exclude)])
        return result

    def mark(self, path: Iterator[MazeLocation]) -> str:
        grid = deepcopy(self._grid)
        for p in path:
            grid[p.row][p.col] = Cell.PATH
        grid[self.start.row][self.start.col] = Cell.START
        grid[self.goal.row][self.goal.col] = Cell.GOAL

        return _grid_as_str(grid)

    def goal_test(self, location: MazeLocation) -> bool:
        return self.goal == location

    def _valid_location(self, cur: MazeLocation, other: MazeLocation, exclude: Optional[Cell]) -> bool:
        """
        Check that "other" is a valid maze location:

        * is within (self) maze bounds
        * is not "cur"
        * is not "exclude"
        """
        if (0 <= other[0] < self.rows and 0 <= other[1] < self.cols) \
                and (cur != other) \
                and self._grid[other[0]][other[1]] != exclude:
            return True
        return False

    def _fill_maze(self) -> None:
        self._grid = [[Cell.EMPTY for _ in range(
            self.cols)] for _ in range(self.rows)]
        for (i, j) in product(range(self.rows), range(self.cols)):
            if uniform(0.0, 1.0) < self.sparseness:
                self._grid[i][j] = Cell.BLOCKED
            else:
                self._grid[i][j] = Cell.EMPTY
        self._grid[self.start.row][self.start.col] = Cell.START
        self._grid[self.goal.row][self.goal.col] = Cell.GOAL

    def __str__(self) -> str:
        return _grid_as_str(self._grid)


def _grid_as_str(grid: List[List[Cell]]) -> str:
    output = []
    for row in grid:
        output.append("".join([c for c in row]))
    return "\n".join(output)


T = TypeVar("T")


@dataclass
class Stack(Generic[T]):
    _container: List[T] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, other: T) -> None:
        self._container.append(other)

    def pop(self) -> T:
        return self._container.pop()


@dataclass
class Queue(Generic[T]):
    _container: Deque[T] = field(default_factory=deque)

    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, other: T) -> None:
        self._container.append(other)

    def pop(self) -> T:
        return self._container.popleft()


@dataclass
class Node(Generic[T]):
    state: T
    parent: Optional[Self] = field(default=None)
    cost: float = field(default=0.0)
    heuristic: float = field(default=0.0)

    def __lt__(self, other: Self) -> bool:
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


def _search_algo(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]], frontier: Stack[Node[T]] | Queue[Node[T]]) -> Optional[Node[T]]:
    """
    Search algorithm. The "frontier" arg makes it either a depth-first or a breath-first algorithm.
    """
    explored: Set[T] = {initial}

    frontier.push(Node(state=initial, parent=None))

    while not frontier.empty:
        cur_node: Node[T] = frontier.pop()
        cur_state: T = cur_node.state
        if goal_test(cur_state):
            return cur_node
        for succ in successors(cur_state):
            if succ in explored:
                continue
            explored.add(succ)
            frontier.push(Node(state=succ, parent=cur_node))
    return None


dfs = partial(_search_algo, frontier=Stack())
bfs = partial(_search_algo, frontier=Queue())


def to_path(node: Node[T]) -> Iterator[T]:
    result: List[T] = [node.state]
    while not node.parent is None:
        node = node.parent
        result.append(node.state)
    return reversed(result)


def main():
    # seed(42)

    m = Maze(cols=11)

    print("----     Maze     ----")
    print(m)
    print("----------------------")

    solution_dfs: Optional[Node[MazeLocation]] = dfs(
        m.start, m.goal_test, m.successors)

    print("---- DFS solution ----")
    match solution_dfs:
        case Node():
            print(m.mark(to_path(solution_dfs)))
        case None:
            print("No DFS solution found.")
    print("----------------------")

    solution_bfs: Optional[Node[MazeLocation]] = bfs(
        m.start, m.goal_test, m.successors)

    print("---- BFS solution ----")
    match solution_bfs:
        case Node():
            print(m.mark(to_path(solution_bfs)))
        case None:
            print("No BFS solution found.")
    print("----------------------")


if __name__ == "__main__":
    main()
