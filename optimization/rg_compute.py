from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


# -----------------------------------------------------------------------------
# Trajectory-driven Rg generation has been intentionally removed from the
# public release. In the original pipeline this module loaded an MD
# trajectory through a domain-specific reader, unwrapped the chain across periodic
# boundaries, and emitted ``rg_data.txt`` (two columns: time in ns, Rg in
# nm) plus a "last frame" Rg snapshot for the live monitor.
#
# To wire your own simulation engine into this BO controller, implement the
# two functions below so they:
#
#   * ``compute_rg_series(traj_path, output_path, dt_fs)`` -- write a
#     two-column text file ``rg_data.txt`` from a finished trajectory.
#   * ``compute_last_frame_rg(traj_path, dt_fs)`` -- return ``(time_ns,
#     rg_nm)`` for the most recent frame, or ``None`` if the file is not yet
#     readable (live monitor calls this every poll).
#
# The fake runner shipped with this repo (``optimization/fake_run.py``)
# bypasses both functions by writing ``rg_data.txt`` directly, so the BO
# controller can be exercised end-to-end without a real MD engine.
# -----------------------------------------------------------------------------


def timestep_to_ns(step: float, dt_fs: float = 10.0) -> float:
    return float(step) * float(dt_fs) * 1.0e-6


def compute_rg_series(
    gsd_file: str | Path,
    output_file: str | Path | None = None,
    dt_fs: float = 10.0,
) -> np.ndarray:
    raise NotImplementedError(
        "Plug your own trajectory loader in here; see the module docstring."
    )


def compute_last_frame_rg(gsd_file: str | Path, dt_fs: float = 10.0) -> tuple[float, float] | None:
    """Best-effort live snapshot. Returning ``None`` is safe -- the monitor
    treats it as 'not yet available' and tries again on the next poll."""
    return None


def ensure_rg_data(run_dir: str | Path, dt_fs: float = 10.0, force: bool = True) -> Path:
    """If a finished simulation already wrote ``rg_data.txt`` (e.g. the fake
    runner), keep it. Otherwise the user's plugged-in ``compute_rg_series``
    is invoked to build one from a trajectory."""
    run_path = Path(run_dir)
    output_file = run_path / "rg_data.txt"
    if output_file.exists() and not force:
        return output_file
    if output_file.exists():
        return output_file
    gsd_file = run_path / "nosol.gsd"
    compute_rg_series(gsd_file, output_file, dt_fs)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rg_data.txt from a finished trajectory.")
    parser.add_argument("run_dir")
    parser.add_argument("--dt-fs", type=float, default=10.0)
    args = parser.parse_args()
    output = ensure_rg_data(Path(args.run_dir), args.dt_fs)
    print(output)


if __name__ == "__main__":
    main()
