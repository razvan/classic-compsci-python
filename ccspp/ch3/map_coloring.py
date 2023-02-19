"""Solve a map coloring problem.
Given the map of Australia's territories and three colors (red, green and blue),
colorize the map such that no two adjacent territories share the same color.
"""
from typing import Dict, List, Set, Optional

from .csp import Constraint, CSP


class MapColoringConstraint(Constraint[str, str]):  # pylint: disable=too-few-public-methods
    """The coloring constraint. Two territories cannot have the same color
    in the given assignment."""

    def __init__(self, place1: str, place2: str) -> None:
        super().__init__({place1, place2})
        self.place1 = place1
        self.place2 = place2

    def satisfied(self, assignment: Dict[str, str]) -> bool:
        if self.place1 not in assignment or self.place2 not in assignment:
            return True
        return assignment[self.place1] != assignment[self.place2]


def main():
    """Run the coloring algorithm."""
    variables: Set[str] = {"Western Australia", "Northern Territory",
                           "South Australia", "Queensland",
                           "New South Wales", "Victoria", "Tasmania"}
    domains: Dict[str, List[str]] = {}
    for variable in variables:
        domains[variable] = ["red", "green", "blue"]

    csp: CSP[str, str] = CSP(variables, domains)

    csp.append_constraint(MapColoringConstraint(
        "Western Australia", "Northern Territory"))
    csp.append_constraint(MapColoringConstraint(
        "Western Australia", "South Australia"))
    csp.append_constraint(MapColoringConstraint(
        "South Australia", "Northern Territory"))
    csp.append_constraint(MapColoringConstraint(
        "Queensland", "Northern Territory"))
    csp.append_constraint(MapColoringConstraint(
        "Queensland", "South Australia"))
    csp.append_constraint(MapColoringConstraint(
        "Queensland", "New South Wales"))
    csp.append_constraint(MapColoringConstraint(
        "New South Wales", "South Australia"))
    csp.append_constraint(MapColoringConstraint("Victoria", "South Australia"))
    csp.append_constraint(MapColoringConstraint("Victoria", "New South Wales"))
    csp.append_constraint(MapColoringConstraint("Victoria", "Tasmania"))

    solution: Optional[Dict[str, str]] = csp.backtracking_search(None)
    if solution is None:
        print("No solution found!")
    else:
        print(solution)


if __name__ == '__main__':
    main()
