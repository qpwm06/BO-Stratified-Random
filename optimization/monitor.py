from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from optimization.common import campaign_dir
from optimization.preprocess import load_sequence_table
from optimization.rg_compute import compute_last_frame_rg

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


def parse_remaining_seconds(text: str) -> int | None:
    """Parse a worker's thermo log line: ``<n_steps> HH:MM:SS ...``."""
    parts = text.split()
    if len(parts) < 2:
        return None
    fields = parts[1].split(":")
    if len(fields) != 3 or not all(field.isdigit() for field in fields):
        return None
    hours, minutes, seconds = (int(field) for field in fields)
    return hours * 3600 + minutes * 60 + seconds


def latest_thermo_remaining_seconds(run_dir: Path, thermo_names: list[str]) -> int | None:
    for name in thermo_names:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            remaining = parse_remaining_seconds(line)
            if remaining is not None:
                return remaining
    return None


def resident_panel_counts(config: dict[str, Any]) -> dict[str, int]:
    """Snapshot the resident pool's progress.

    ``thermo_md_jobs`` counts RUNNING tasks that have produced a parseable
    log line; ``updated_jobs`` is the subset whose remaining-time estimate
    decreased compared to the previous poll (i.e. they really are making
    progress, not stuck on the same step).
    """
    queue_name = config.get("resident_worker", {}).get("queue_dir", "resident_queue")
    queue_dir = campaign_dir(config) / queue_name
    tasks_dir = queue_dir / "tasks"
    if not tasks_dir.exists():
        return {"updated_jobs": 0, "thermo_md_jobs": 0}
    thermo_names = config.get("resident_worker", {}).get(
        "startup_watchdog_files", ["thermo_md_log.txt"]
    )
    progress_path = queue_dir / "panel_progress.json"
    try:
        previous = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        previous = {}
    current: dict[str, int] = {}
    updated_jobs = 0
    thermo_md_jobs = 0
    for task_path in sorted(tasks_dir.glob("*.json")):
        try:
            task = pd.read_json(task_path, typ="series").to_dict()
        except ValueError:
            continue
        if task.get("status") != "RUNNING":
            continue
        run_dir = Path(str(task.get("run_dir", "")))
        remaining = latest_thermo_remaining_seconds(run_dir, [str(name) for name in thermo_names])
        if remaining is not None:
            thermo_md_jobs += 1
            task_id = str(task.get("task_id") or task_path.stem)
            current[task_id] = remaining
            old_remaining = previous.get(task_id)
            if isinstance(old_remaining, (int, float)) and remaining < old_remaining:
                updated_jobs += 1
    try:
        progress_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass
    return {"updated_jobs": updated_jobs, "thermo_md_jobs": thermo_md_jobs}


def load_reference_rg(config: dict[str, Any]) -> dict[str, float]:
    table = load_sequence_table(config)
    rg_col = "exp_rg_nm" if "exp_rg_nm" in table.columns else "Rg"
    return dict(zip(table["label"].astype(str), table[rg_col].astype(float)))


def compute_temp_mae(config: dict[str, Any], candidate_dir: str | Path) -> tuple[float | None, int, int]:
    """Best-effort live MAE from the most recent dumped frame of each sequence.

    Discovers labels from the candidate's runfile/ subdirectories so the
    stratified controller (which gives each candidate its own subset) and
    the vanilla baseline (single ``selected_sequences`` list) both work.
    """
    refs = load_reference_rg(config)
    root = Path(candidate_dir)
    runfile_root = root / "runfile"
    if runfile_root.exists():
        labels = sorted(p.name for p in runfile_root.iterdir() if p.is_dir())
    else:
        labels = list(config.get("selected_sequences", []))
    errors = []
    for label in labels:
        latest = compute_last_frame_rg(root / "runfile" / label / "nosol.gsd")
        if latest is None or label not in refs:
            continue
        _, latest_rg = latest
        errors.append(abs(latest_rg - refs[label]))
    if not errors:
        return None, 0, len(labels)
    return float(sum(errors) / len(errors)), len(errors), len(labels)


def build_status_rows(config: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    now = datetime.now().isoformat(timespec="seconds")
    panel_counts = resident_panel_counts(config)
    for obs in state.get("observations", []):
        rows.append(
            {
                "timestamp": now,
                "trial_key": obs.get("trial_key"),
                "candidate_index": obs.get("candidate_index"),
                "status": obs.get("status"),
                "source": obs.get("source"),
                "loss": obs.get("loss"),
                "candidate_dir": "",
                "jobs_done": "",
                "jobs_total": "",
                **panel_counts,
            }
        )
    # Stratified BO stores running candidates under "candidates"; the vanilla
    # baseline used "running". Accept both.
    running_pool = state.get("running") or {
        k: c for k, c in state.get("candidates", {}).items() if c.get("status") == "RUNNING"
    }
    for key, item in running_pool.items():
        statuses = item.get("job_status", {}) or {}
        done = sum(1 for status in statuses.values() if status == "COMPLETED")
        temp_mae, temp_n, temp_total = compute_temp_mae(config, item.get("candidate_dir", ""))
        rows.append(
            {
                "timestamp": now,
                "trial_key": key,
                "candidate_index": item.get("candidate_index"),
                "status": item.get("status", "RUNNING"),
                "source": "simulation",
                "loss": "",
                "candidate_dir": item.get("candidate_dir", ""),
                "jobs_done": done,
                "jobs_total": len(statuses) if statuses else len(item.get("jobs", {})),
                "temp_last_frame_mae_nm": temp_mae if temp_mae is not None else "",
                "temp_rg_seen": temp_n,
                "temp_rg_total": temp_total,
                **panel_counts,
            }
        )
    for item in state.get("failed", []):
        rows.append(
            {
                "timestamp": now,
                "trial_key": f"candidate_{int(item.get('candidate_index', -1)):04d}",
                "candidate_index": item.get("candidate_index"),
                "status": "FAILED",
                "source": "simulation",
                "loss": "",
                "candidate_dir": item.get("candidate_dir", ""),
                "jobs_done": "",
                "jobs_total": "",
                **panel_counts,
            }
        )
    return rows


def write_status_csv(config: dict[str, Any], state: dict[str, Any]) -> Path:
    root = campaign_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_status_rows(config, state)
    df = pd.DataFrame(rows)
    path = root / "status.csv"
    df.to_csv(path, index=False)
    return path


def refresh_performance_plot(config: dict[str, Any], state: dict[str, Any]) -> Path | None:
    """BO loss curve with shaded Sobol / BO regions and a top-mounted legend.

    The figure has two background bands:

    * Sobol exploration phase: ``[0, n_initial_sobol)``.
    * BO optimization phase  : ``[n_initial_sobol, +∞)``.

    Each band is annotated with a label so the reader can immediately see
    where Sobol stops and BoTorch takes over. The legend is moved above
    the axes (``ncol=3``) so the dots and best-so-far line are never
    occluded.
    """
    observations = list(state.get("observations", []))
    root = campaign_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    if not observations:
        return None

    df = pd.DataFrame(
        {
            "candidate_index": [
                obs["candidate_index"] if obs.get("candidate_index") is not None else -index - 1
                for index, obs in enumerate(observations)
            ],
            "loss": [obs["loss"] for obs in observations],
            "source": [str(obs.get("source", "unknown")).lower() for obs in observations],
        }
    ).sort_values("candidate_index")
    df["best_loss"] = df["loss"].cummin()

    import matplotlib.pyplot as plt

    n_initial_sobol = int(config.get("bo", {}).get("n_initial_sobol", 0))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))

    # Background bands.
    x_max = float(df["candidate_index"].max()) + 1
    band_top = float(df["loss"].max())
    band_bot = float(df["loss"].min())
    pad = 0.1 * (band_top - band_bot if band_top > band_bot else band_top + 1.0)
    ymin, ymax = band_bot - pad, band_top + 1.6 * pad
    if n_initial_sobol > 0:
        ax.axvspan(-0.5, n_initial_sobol - 0.5, facecolor="#ececec", zorder=0)
    if n_initial_sobol < x_max:
        ax.axvspan(n_initial_sobol - 0.5, x_max - 0.5, facecolor="#e8f1fb", zorder=0)

    # Source markers. We accept multiple legacy source tags and remap them.
    sobol_mask = df["source"].isin(["sobol", "warm_start", "warm_start_alpha"])
    bo_mask = df["source"].isin(["botorch", "qnei", "ax", "qlognei", "simulation"])
    other_mask = ~(sobol_mask | bo_mask)

    if sobol_mask.any():
        ax.scatter(df.loc[sobol_mask, "candidate_index"], df.loc[sobol_mask, "loss"],
                   c="#555555", marker="o", s=22, label="Sobol obs.", zorder=2)
    if bo_mask.any():
        ax.scatter(df.loc[bo_mask, "candidate_index"], df.loc[bo_mask, "loss"],
                   facecolors="none", edgecolors="#1f77b4", marker="o", s=22,
                   label="BoTorch obs.", zorder=2)
    if other_mask.any():
        ax.scatter(df.loc[other_mask, "candidate_index"], df.loc[other_mask, "loss"],
                   facecolors="none", edgecolors="#bbbbbb", marker="o", s=18,
                   label="other obs.", zorder=2)

    # In-flight estimates from currently-running candidates.
    temp_rows = []
    for item in state.get("running", {}).values():
        temp_mae, temp_n, _ = compute_temp_mae(config, item.get("candidate_dir", ""))
        if temp_mae is not None:
            temp_rows.append({"candidate_index": item.get("candidate_index"),
                              "temp_mae_proxy": temp_mae})
    if temp_rows:
        temp_df = pd.DataFrame(temp_rows)
        ax.scatter(temp_df["candidate_index"], temp_df["temp_mae_proxy"],
                   c="#d62728", marker="x", s=30, label="ongoing", zorder=3)

    ax.plot(df["candidate_index"], df["best_loss"], color="black", linewidth=1.4,
            label="best so far", zorder=4)

    # Region labels at the top of each band.
    if n_initial_sobol > 0:
        ax.text((n_initial_sobol - 0.5) / 2, ymax - 0.05 * (ymax - ymin),
                f"Sobol exploration (n={n_initial_sobol})",
                ha="center", va="top", fontsize=8.5, color="#444444", zorder=1)
    if n_initial_sobol < x_max:
        ax.text((n_initial_sobol - 0.5 + x_max) / 2, ymax - 0.05 * (ymax - ymin),
                "BoTorch BO (qLogNEI)",
                ha="center", va="top", fontsize=8.5, color="#1f4f80", zorder=1)

    ax.set_xlabel("candidate index")
    ax.set_ylabel("loss")
    ax.set_xlim(-0.5, x_max - 0.5)
    ax.set_ylim(ymin, ymax)
    # Top-mounted legend, ncol=3, outside axes so it never covers data.
    ax.legend(frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0.0,
              fontsize=8.5)
    fig.tight_layout()
    path = root / "performance.png"
    tmp_path = root / ".performance.tmp.png"
    fig.savefig(tmp_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    tmp_path.replace(path)
    return path


def refresh_monitor_outputs(config: dict[str, Any], state: dict[str, Any]) -> None:
    status_path = write_status_csv(config, state)
    plot_path = refresh_performance_plot(config, state)
    print(f"[monitor] status -> {status_path}", flush=True)
    if plot_path is not None:
        print(f"[monitor] performance -> {plot_path}", flush=True)
