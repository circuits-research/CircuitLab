from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    launcher: str  # "torchrun", "slurm", "single"


def _infer_from_torchrun() -> Optional[DistributedContext]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return DistributedContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_distributed=(world_size > 1),
            launcher="torchrun",
        )
    return None


def _infer_from_slurm() -> Optional[DistributedContext]:
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        return DistributedContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_distributed=(world_size > 1),
            launcher="slurm",
        )
    return None


def infer_distributed_context() -> DistributedContext:
    ctx = _infer_from_torchrun() or _infer_from_slurm()
    if ctx is not None:
        return ctx
    return DistributedContext(rank=0, world_size=1, local_rank=0, is_distributed=False, launcher="single")


def _ensure_master_env_for_single_node_default() -> None:
    """
    For torchrun, MASTER_ADDR/MASTER_PORT are typically set.
    For slurm, they may or may not be. If missing, we set conservative defaults.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def init_distributed(
    *,
    backend: str = "nccl",
    timeout_minutes: int = 30,
    set_cuda_device: bool = True,
) -> DistributedContext:
    """
    Initializes torch.distributed if needed. Safe to call in single-process mode.

    Returns a DistributedContext.
    """
    ctx = infer_distributed_context()

    if set_cuda_device and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank)

    if ctx.is_distributed:
        if not dist.is_initialized():
            # Ensure required rendezvous env vars exist at least for single-node patterns.
            # Multi-node slurm setups usually provide these externally.
            _ensure_master_env_for_single_node_default()

            dist.init_process_group(
                backend=backend,
                rank=ctx.rank,
                world_size=ctx.world_size,
                timeout=timedelta(minutes=timeout_minutes),
            )
    return ctx


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_once(msg: str) -> None:
    if is_rank0():
        print(msg, flush=True)


def host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"
