from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .distributed import barrier_if_distributed, is_rank0


def ensure_dirs(paths: Sequence[Path], *, exist_ok: bool = True, barrier: bool = True) -> None:
    """
    Create directories only on rank 0, then optionally barrier.

    This prevents N ranks racing to mkdir on shared filesystems.
    """
    if is_rank0():
        for p in paths:
            p.mkdir(parents=True, exist_ok=exist_ok)

    if barrier:
        barrier_if_distributed()