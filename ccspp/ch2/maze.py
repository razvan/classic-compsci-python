#!/usr/bin/env python3

# this is needed for Node definition which references it's self.
from __future__ import annotations
from functools import partial
from collections import deque
from enum import StrEnum
from typing import Deque, Dict, Iterator, List, Optional, Generic, TypeVar, Callable, Set
from dataclasses import dataclass, field
from itertools import product
from random import uniform, seed
from copy import deepcopy
from heapq import heappush, heappop
from math import sqrt

T = TypeVar("T")


class Cell(StrEnum):
    EMPTY = " "
    BLOCKED = "X"
    START = "S"
    GOAL = "G"
    PATH = "*"


@dataclass(frozen=True, slots=True)
class MazeLocation:
    row: int
    col: int


@dataclass(slots=True)
class Maze:
    sparseness: float = field(default=0.2)
    rows: int = field(default=10)
    cols: int = field(default=10)
    start: MazeLocation = field(default=MazeLocation(0, 0))
    goal: MazeLocation = field(default=MazeLocation(9, 9))
    _grid: List[List[Cell]] = field(repr=False, init=False)

    def __post_init__(self) -> None:
        self._grid = Maze._fill_maze(self.rows, self.cols,
                                     self.sparseness, self.start, self.goal)

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

        return Maze._grid_as_str(grid)

    def goal_test(self, location: MazeLocation) -> bool:
        return self.goal == location

    def _valid_location(self, cur: MazeLocation, other: MazeLocation, exclude: Optional[Cell]) -> bool:
        """
        Check that "other" is a valid maze location:

        * is within (self) maze bounds
        * is not "cur"
        * is not "exclude"
        """
        if (0 <= other.row < self.rows and 0 <= other.col < self.cols) \
                and (cur != other) \
                and self._grid[other.row][other.col] != exclude:
            return True
        return False

    def __str__(self) -> str:
        return Maze._grid_as_str(self._grid)

    @classmethod
    def _fill_maze(cls, rows: int, cols: int, sparseness: float, start: MazeLocation, goal: MazeLocation) -> List[List[Cell]]:
        _grid = [[Cell.EMPTY for _ in range(
            cols)] for _ in range(rows)]
        for (i, j) in product(range(rows), range(cols)):
            if uniform(0.0, 1.0) < sparseness:
                _grid[i][j] = Cell.BLOCKED
            else:
                _grid[i][j] = Cell.EMPTY
        _grid[start.row][start.col] = Cell.START
        _grid[goal.row][goal.col] = Cell.GOAL
        return _grid

    @classmethod
    def _grid_as_str(cls, grid: List[List[Cell]]) -> str:
        output = []
        for row in grid:
            output.append("".join([c for c in row]))
        return "\n".join(output)


@dataclass(frozen=True, slots=True)
class Stack(Generic[T]):
    _container: List[T] = field(default_factory=list)

    @ property
    def empty(self) -> bool:
        return not self._container

    def push(self, other: T) -> None:
        self._container.append(other)

    def pop(self) -> T:
        return self._container.pop()


@dataclass(frozen=True, slots=True)
class Queue(Generic[T]):
    _container: Deque[T] = field(default_factory=deque)

    @ property
    def empty(self) -> bool:
        return not self._container

    def push(self, other: T) -> None:
        self._container.append(other)

    def pop(self) -> T:
        return self._container.popleft()


@ dataclass(slots=True, frozen=True)
class PriorityQueue(Generic[T]):
    container: List[T] = field(default_factory=list)

    @ property
    def empty(self) -> bool:
        return not self.container

    def push(self, o: T) -> None:
        heappush(self.container, o)

    def pop(self) -> T:
        return heappop(self.container)


@dataclass(frozen=True, slots=True)
class Node(Generic[T]):
    state: T
    parent: Optional[Node[T]] = field(default=None)
    cost: float = field(default=0.0)
    heuristic: float = field(default=0.0)

    def __lt__(self, other: Node[T]) -> bool:
        """ Needed in particular for ordreing nodes in a PriorityQeue."""
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


def _astar(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]], frontier: PriorityQueue[Node[T]], heuristic: Callable[[T], float]) -> Optional[Node[T]]:
    """
    A* search algorithm
    """
    explored: Dict[T, float] = {initial: 0.0}  # node -> cost

    frontier.push(Node(state=initial, parent=None,
                  heuristic=heuristic(initial)))

    while not frontier.empty:
        cur_node: Node[T] = frontier.pop()
        cur_state: T = cur_node.state
        if goal_test(cur_state):
            return cur_node
        for succ in successors(cur_state):
            new_cost: float = cur_node.cost + 1
            if succ not in explored or explored[succ] > new_cost:
                explored[succ] = new_cost
                frontier.push(Node(state=succ, parent=cur_node,
                              cost=new_cost, heuristic=heuristic(succ)))
    return None


def dist_euclidian(_from: MazeLocation, to: MazeLocation) -> float:
    xdist: int = to.row - _from.row
    ydist: int = to.col - _from.col
    return sqrt((xdist * xdist) + (ydist * ydist))


def dist_manhattan(_from: MazeLocation, to: MazeLocation) -> float:
    xdist: int = abs(to.row - _from.row)
    ydist: int = abs(to.col - _from.col)
    return float(xdist + ydist)


"""
A type alias for search algoritms. This type is basically the signature of the `_search_algo()`
function except for the `frontier`  parameter which is curried using `partial()`
"""
TSearchAlgorithm = Callable[[T, Callable[[T], bool],
                             Callable[[T], List[T]]], Optional[Node[T]]]

dfs: TSearchAlgorithm = partial(_search_algo, frontier=Stack())
bfs: TSearchAlgorithm = partial(_search_algo, frontier=Queue())


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

    manhattan = partial(dist_manhattan, to=m.goal)
    astar = partial(_astar, frontier=PriorityQueue(), heuristic=manhattan)
    solution_astar = astar(m.start, m.goal_test, m.successors)
    print("---- A* solution ----")
    match solution_astar:
        case Node():
            print(m.mark(to_path(solution_astar)))
        case None:
            print("No A* solution found.")
    print("----------------------")


if __name__ == "__main__":
    main()
