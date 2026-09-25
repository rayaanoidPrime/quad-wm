"""Process/run provenance recorded with every experiment run."""

from __future__ import annotations

import os
import platform
import socket
import sys


def run_metadata(config: dict | None = None) -> dict[str, str]:
    """Common provenance fields shared by every entry point."""
    config = config or {}
    return {
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "config_path": config.get("config_path", "unknown"),
        "git_commit": os.environ.get("QUADWM_GIT_COMMIT", "unknown"),
        "visible_devices": os.environ.get(
            "ROCR_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        ),
    }