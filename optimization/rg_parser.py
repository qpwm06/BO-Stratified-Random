from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd


def parse_rg_file(path: str | Path, burnin_fraction: float = 0.5, burnin_ns: float | None = None) -> dict[str, float]:
    """Parse an ``rg_data.txt`` (two-column ``time(ns) Rg(nm)``).

    Drops a leading burn-in window (either as a fraction of frames or an
    absolute time in nanoseconds) and returns mean / standard error / span.
    """
    rg_path = Path(path)
    if not rg_path.exists():
        raise FileNotFoundError(f"Missing Rg file: {rg_path}")
    data = np.loadtxt(rg_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Rg file needs at least two columns: {rg_path}")
    if burnin_ns is None:
        start = int(len(data) * float(burnin_fraction))
    else:
        start_time = float(data[0, 0]) + float(burnin_ns)
        start = int(np.searchsorted(data[:, 0], start_time, side="left"))
    values = data[start:, 1]
    if len(values) == 0:
        raise ValueError(f"No Rg samples after burn-in: {rg_path}")
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "n_frames": int(len(values)),
        "sim_rg_nm": float(values.mean()),
        "sim_se_nm": se,
        "first_time_ns": float(data[start, 0]),
        "last_time_ns": float(data[-1, 0]),
    }


def parse_candidate(
    candidate_dir: str | Path,
    labels: list[str],
    burnin_fraction: float,
    burnin_ns: float | None = None,
) -> pd.DataFrame:
    rows = []
    root = Path(candidate_dir)
    for label in labels:
        stats = parse_rg_file(root / "runfile" / label / "rg_data.txt", burnin_fraction, burnin_ns)
        stats["label"] = label
        rows.append(stats)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse rg_data.txt and report mean / SE.")
    parser.add_argument("rg_file")
    parser.add_argument("--burnin-fraction", type=float, default=0.5)
    parser.add_argument("--burnin-ns", type=float, default=None)
    args = parser.parse_args()
    print(parse_rg_file(args.rg_file, args.burnin_fraction, args.burnin_ns))


if __name__ == "__main__":
    main()
