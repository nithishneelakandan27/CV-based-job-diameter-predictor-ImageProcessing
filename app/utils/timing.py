"""Small timing helpers used by the image pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator


@contextmanager
def elapsed_milliseconds() -> Iterator[Callable[[], float]]:
    """Yield a callable that reports elapsed wall-clock milliseconds."""

    start = perf_counter()
    yield lambda: (perf_counter() - start) * 1000.0
