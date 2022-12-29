#
# Run tests with:
#
# python -m doctest -v fibonacci.py

from functools import lru_cache
from typing import Generator


@lru_cache(maxsize=None)
def fib_rec(n: int) -> int:
    """
    >>> fib_rec(-1)
    Traceback (most recent call last):
    ...
    ValueError: Invalid value -1
    >>> fib_rec(2)
    1
    >>> fib_rec(5)
    5
    >>> fib_rec(50)
    12586269025
    """
    if n < 0:
        raise ValueError(f"Invalid value {n}")
    elif n < 2:
        return n
    else:
        return fib_rec(n-1) + fib_rec(n-2)


def fib_iter(n: int) -> int:
    """
    >>> fib_iter(5)
    5
    >>> fib_iter(50)
    12586269025
    """
    if n < 0:
        raise ValueError(f"Invalid value {n}")
    last: int = 0
    next: int = 1
    for _ in range(1, n):
        last, next = next, last + next
    return next


def fib_gen(n: int) -> Generator[int, None, None]:
    """
    >>> list(fib_gen(5)) # doctest: +ELLIPSIS
    [0, 1, ..., 5]
    >>> list(fib_gen(50))  # doctest: +ELLIPSIS
    [0, 1, ..., 12586269025]
    """
    if n < 0:
        raise ValueError(f"Invalid value {n}")
    yield 0
    if n > 1:
        yield 1
    last: int = 0
    next: int = 1
    for _ in range(1, n):
        last, next = next, last + next
        yield next
