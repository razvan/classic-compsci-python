"""Constraint solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, List, Dict, Set, Iterable, Optional

V = TypeVar("V")  # variable type
D = TypeVar("D")  # domain type


class Constraint(Generic[V, D], ABC):  # pylint: disable=too-few-public-methods
    """Base class for all constraints."""

    def __init__(self, variables: Set[V]) -> None:
        self.variables: Set[V] = variables

    @abstractmethod
    def satisfied(self, assignment: Dict[V, D]) -> bool:
        """Verify that the constraint satisfies the assignment."""


def _validate_empty_difference(set1: Set[V], set2: Iterable[V], /):
    diff = set1.difference(set2)
    if diff:
        raise LookupError(f"Unknown variables: [{diff}]")


@dataclass(frozen=True)
class CSP(Generic[V, D]):
    """A constraint satisfaction problem has variables of type V
    that have ranges of values known as domains D and constraints
    that determine if a variable's domain selection is satisfied."""
    variables: Set[V] = field()
    domains: Dict[V, List[D]] = field()
    constraints: Dict[V, List[Constraint[V, D]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_empty_difference(self.variables, self.domains.keys())

    def append_constraint(self, constraint: Constraint[V, D]) -> None:
        """Update constraints"""
        _validate_empty_difference(set(constraint.variables), self.variables)
        for convar in constraint.variables:
            if convar in self.constraints:
                self.constraints[convar].append(constraint)
            else:
                self.constraints[convar] = [constraint]

    def is_consistent(self, variable: V, assignment: Dict[V, D]) -> bool:
        """Verify that all constraints of variable are satisfied by the assignment."""
        return all((c.satisfied(assignment) for c in self.constraints[variable]))

    def backtracking_search(self, assignment: Optional[Dict[V, D]] = None) -> Optional[Dict[V, D]]:
        """Use backtracking to search for an assignment that satisfies all constraints
        for all variables."""
        if not assignment:
            assignment = {}

        if len(assignment) == len(self.variables):
            return assignment

        unassigned = self.variables.difference(assignment.keys())
        first: V = next(iter(unassigned))
        for domain in self.domains[first]:
            local_assignment = assignment.copy()
            local_assignment[first] = domain
            if self.is_consistent(first, local_assignment):
                result: Optional[Dict[V, D]] = self.backtracking_search(
                    local_assignment)
                if result:
                    return result
        return None
