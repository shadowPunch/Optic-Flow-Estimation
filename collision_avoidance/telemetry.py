"""Weights & Biases run management shared by every entry point.

Set COLLISION_WANDB=0 to run offline without tracking. When tracking is on and
W&B cannot be initialised, we fail loudly instead of running untracked.
"""

import os
from dataclasses import asdict, is_dataclass

WANDB_PROJECT = os.environ.get("COLLISION_WANDB_PROJECT", "kria-collision-avoidance")


def wandb_enabled() -> bool:
    return os.environ.get("COLLISION_WANDB", "1") != "0"


def start_run(job_type: str, config, tags: list[str] | None = None, name: str | None = None, notes: str | None = None):
    """Return an active wandb run, or None when tracking is disabled."""
    if not wandb_enabled():
        print("[telemetry] COLLISION_WANDB=0 -> W&B tracking disabled for this run")
        return None
    import wandb

    cfg = asdict(config) if is_dataclass(config) else dict(config)
    try:
        return wandb.init(project=WANDB_PROJECT, job_type=job_type, config=cfg, tags=tags, name=name, notes=notes)
    except Exception as exc:  # auth / network problems
        raise RuntimeError(
            f"W&B initialisation failed ({exc}). Fix credentials/network or set COLLISION_WANDB=0 to run untracked."
        ) from exc
