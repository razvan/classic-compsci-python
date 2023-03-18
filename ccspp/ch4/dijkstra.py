"""
This module contains a bidirectional graph implementation that uses adjacency lists
and a shortest path finding algorithm (Dijkstra).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Callable, Generator, Iterable, List, Dict, TypeVar, Generic, Optional

Vertex = TypeVar("Vertex")


@dataclass(frozen=True)
class _IndexedEdge:
    """An edge where the indices point to the two vertices in the graph's vertices list.
    Not part of the public api.
    """
    from_i: int = 0
    to_i: int = 0
    weight: float = 0.0

    def reverse(self) -> _IndexedEdge:
        """Reverse edge"""
        return _IndexedEdge(self.to_i, self.from_i, self.weight)


class GraphBuilder(Generic[Vertex]):
    """Graph builder."""
    _split_pred: Callable[[_IndexedEdge, _IndexedEdge], bool] = lambda var1, var2: var1.from_i != var2.from_i

    def __init__(self) -> None:
        self._indices: Dict[Vertex, int] = {}
        self._edges: List[_IndexedEdge] = []

    def with_edge(self, from_v: Vertex, to_v: Vertex, weight: float) -> GraphBuilder:
        """Add edge from the vertex from_v to the vertex to_v with the given weight.
        The reverse edge is also added automatically.
        """
        index_from = self._indices.get(from_v)
        if index_from is None:
            index_from = len(self._indices)
            self._indices[from_v] = index_from
        index_to = self._indices.get(to_v)
        if index_to is None:
            index_to = len(self._indices)
            self._indices[to_v] = index_to
        ie = _IndexedEdge(index_from, index_to, weight)
        self._edges.extend([ie, ie.reverse()])
        return self

    def build(self) -> Graph:
        return Graph(
            vertices=[v for (v, _) in sorted(
                self._indices.items(), key=lambda t: t[1])],
            edges=list(partition(
                sorted(self._edges, key=lambda e: e.from_i), pred=GraphBuilder._split_pred))
        )


@dataclass(frozen=True)
class Graph(Generic[Vertex]):
    vertices: List[Vertex] = field(default_factory=list)
    edges: List[List[_IndexedEdge]] = field(default_factory=list)

    def shortest_paths(self, start: Vertex):
        first: int = self.vertices.index(start)
        dist: List[float] = [float("inf")] * len(self.vertices)
        dist[first] = 0.0
        paths: Dict[int, _IndexedEdge] = {}
        priority_queue: List[DijkstraNode] = []

        heappush(priority_queue, DijkstraNode(first, 0))
        while priority_queue:
            from_i: int = heappop(priority_queue).vertex
            from_dist = dist[from_i]
            for edge in self.edges[from_i]:
                to_dist = dist[edge.to_i]
                if to_dist > edge.weight + from_dist:
                    dist[edge.to_i] = edge.weight + from_dist
                    paths[edge.to_i] = edge
                    heappush(priority_queue, DijkstraNode(
                        edge.to_i, edge.weight + from_dist))
        return dist, paths


@dataclass(frozen=True, order=True)
class DijkstraNode:
    vertex: int = field(compare=False, default=0)
    dist: float = field(default=0.0)


T = TypeVar("T")


def partition(iterable: Iterable[T], /, *, pred: Optional[Callable[[T, T], bool]] = None) -> Generator[
    List[T], None, None]:
    """Partition elements based on pred"""
    result: List[T] = []
    it = iter(iterable)
    while True:
        try:
            elem = next(it)
            if not pred:
                def pred(a, b): return a != b
            if not result:
                result = [elem]
            else:
                if pred(result[0], elem):
                    yield result
                    result = [elem]
                else:
                    result.append(elem)
        except StopIteration:
            yield result
            return


if __name__ == '__main__':
    graph = GraphBuilder() \
        .with_edge("Seattle", "Chicago", 1737) \
        .with_edge("Seattle", "San Francisco", 678) \
        .with_edge("San Francisco", "Riverside", 386) \
        .with_edge("San Francisco", "Los Angeles", 348) \
        .with_edge("Los Angeles", "Riverside", 50) \
        .with_edge("Los Angeles", "Phoenix", 357) \
        .with_edge("Riverside", "Phoenix", 307) \
        .with_edge("Riverside", "Chicago", 1704) \
        .with_edge("Phoenix", "Dallas", 887) \
        .with_edge("Phoenix", "Houston", 1015) \
        .with_edge("Dallas", "Chicago", 805) \
        .with_edge("Dallas", "Atlanta", 721) \
        .with_edge("Dallas", "Houston", 225) \
        .with_edge("Houston", "Atlanta", 702) \
        .with_edge("Houston", "Miami", 968) \
        .with_edge("Atlanta", "Chicago", 588) \
        .with_edge("Atlanta", "Washington", 543) \
        .with_edge("Atlanta", "Miami", 604) \
        .with_edge("Miami", "Washington", 923) \
        .with_edge("Chicago", "Detroit", 238) \
        .with_edge("Detroit", "Boston", 613) \
        .with_edge("Detroit", "Washington", 396) \
        .with_edge("Detroit", "New York", 482) \
        .with_edge("Boston", "New York", 190) \
        .with_edge("New York", "Philadelphia", 81) \
        .build()

    distances, path_dict = graph.shortest_paths("Seattle")
    print(distances)
    print(path_dict)
