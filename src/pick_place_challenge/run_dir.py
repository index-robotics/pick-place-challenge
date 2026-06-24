"""Timestamped run directories, mirroring mechacarpal's ``exp_local`` layout.

Each train/eval invocation gets its own folder

    exp_local/<YYYY.MM.DD>/<HHMMSS>_<entry>/

so checkpoints, configs, and metrics are archived per run instead of being
overwritten. Small on purpose — no Hydra, just a path + a JSON writer.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def new_run_dir(entry: str, root: str = "exp_local") -> Path:
    """Create and return ``exp_local/<date>/<time>_<entry>/``."""
    now = datetime.now()
    run = Path(root) / now.strftime("%Y.%m.%d") / f"{now.strftime('%H%M%S')}_{entry}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_json(run: Path, name: str, data: dict) -> None:
    (run / f"{name}.json").write_text(json.dumps(data, indent=2))
