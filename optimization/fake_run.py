"""Drop-in replacement for the user's MD ``production.py``.

Useful for end-to-end smoke tests of the BO controller without a real
simulation engine: reads ``params.json`` from the parent ``candidate_*``
directory, derives a deterministic per-sequence Rg from the parameter
vector, and writes a ``rg_data.txt`` plus a placeholder ``nosol.gsd``-named
marker so the rest of the pipeline (parser, loss, monitor) keeps working.

Configure ``paths.simulation_python`` to ``python`` and arrange for the
SBATCH script to invoke ``python optimization/fake_run.py`` (or set
``simulation_python`` to a thin wrapper script that does so). See the
README's quickstart section for the exact wiring.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path


def deterministic_rg(label: str, parameters: dict[str, float]) -> tuple[float, float]:
    """Return (mean_rg_nm, noise_amplitude) deterministic in (label, params)."""
    seed = abs(hash((label, tuple(sorted(parameters.items())))))
    rng = random.Random(seed)
    mean_param = sum(parameters.values()) / max(len(parameters), 1)
    rg = 1.5 + 1.5 * mean_param + 0.25 * rng.uniform(-1.0, 1.0)
    noise = 0.05 + 0.05 * rng.uniform(0.0, 1.0)
    return rg, noise


def write_fake_rg(run_dir: Path, label: str, parameters: dict[str, float]) -> Path:
    rg_mean, noise = deterministic_rg(label, parameters)
    seed = abs(hash((label, tuple(sorted(parameters.items())))) ) % (2**31)
    rng = random.Random(seed)
    n_frames = 200
    dt_ns = 0.05
    lines = ["# Time(ns) Rg(nm)\n"]
    for index in range(n_frames):
        time_ns = (index + 1) * dt_ns
        rg = rg_mean + noise * (rng.random() - 0.5) * 2 * math.exp(-index / 60.0)
        lines.append(f"{time_ns:.6f} {rg:.6f}\n")
    out = run_dir / "rg_data.txt"
    out.write_text("".join(lines), encoding="utf-8")
    (run_dir / "nosol.gsd").write_bytes(b"FAKE_TRAJ_MARKER")
    return out


def main() -> int:
    run_dir = Path.cwd()
    candidate_dir = run_dir.parent.parent  # runfile/<label>/ -> candidate
    label = run_dir.name
    params_path = candidate_dir / "params.json"
    if not params_path.exists():
        print(f"[fake_run] missing params.json at {params_path}", file=sys.stderr)
        return 1
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    parameters = payload.get("lambda_pw") or payload.get("parameters") or {}
    out = write_fake_rg(run_dir, label, parameters)
    print(f"[fake_run] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
