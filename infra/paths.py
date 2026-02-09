from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

def get_user() -> str:
    # Prefer explicit USER, fallback to common alternatives.
    return (
        os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or os.environ.get("USERNAME")
        or "unknown"
    )

def get_fast_base() -> Path:
    """
    Returns the base fast path (cluster-specific).
    Priority: CLT_FAST, FAST, SCRATCH, TMPDIR, else /scratch.
    """
    base = (
        os.environ.get("CLT_FAST")
        or os.environ.get("FAST")
        or os.environ.get("SCRATCH")
        or os.environ.get("TMPDIR")
        or "/scratch"
    )
    return Path(base)

def get_fast_root(project_name: str = "clt") -> Path:
    """
    Returns a stable project root inside fast storage:
      <fast_base>/<user>/<project_name>
    """
    return get_fast_base() / get_user() / project_name


@dataclass(frozen=True)
class RunPaths:
    fast_root: Path
    activations_root: Path
    checkpoints_root: Path

def make_run_paths(
    *,
    model: str,
    project_name: str = "clt",
    run_id: Optional[str] = None,
) -> RunPaths:
    """
    Standardizes your fast filesystem layout.

    Layout:
      <fast_root>/
        activations/<model>[/<run_id>]
        checkpoints/<model>[/<run_id>]
    """
    fast_root = get_fast_root(project_name=project_name)

    def maybe_run(p: Path) -> Path:
        return p / run_id if run_id else p

    base = fast_root
    activ = maybe_run(base / "activations" / model)
    ckpt = maybe_run(base / "checkpoints" / model)

    return RunPaths(
        fast_root=fast_root,
        activations_root=activ,
        checkpoints_root=ckpt,
    )