#!/usr/bin/env python3

def leibnitz_pi(n: int) -> float:
    """
    Compute PI using Leibnitz formula: 4/1 - 4/3 + 4/5 - 4/7 + ...

    >>> "{:.9f}".format(leibnitz_pi(1000))
    '3.140592654'
    """
    pi: float = 0.0
    operation: float = 1.0
    denominator: float = 1.0
    for _ in range(n):
        pi += operation * 4.0/denominator
        denominator += 2
        operation *= -1
    return pi
