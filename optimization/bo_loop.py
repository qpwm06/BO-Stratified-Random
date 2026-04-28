"""Asynchronous BO controller with stratified-random subsampling.

Outer loop:

  1. Refresh status of every running candidate (via Slurm or the resident
     queue), collect per-sequence Rg, finalise loss = fixed_weight *
     fixed_MAE + random_weight * random_MAE.
  2. Persist ``state.json``, ``observations.csv``, ``status.csv``,
     ``performance.png``.
  3. Top up the running set: for each free slot, propose a candidate
     (Sobol for the first ``n_initial_sobol`` trials, then BoTorch via
     Ax), pick this candidate's stratified-random subset of sequences
     (fixed anchor + Rg-stratified random draw — see
     ``optimization.sampling``), stage runfiles, and submit.
  4. If the launcher is ``resident``, ensure the worker pool stays at
     its target size.
  5. Sleep ``slurm.poll_seconds`` and repeat.

Early stop: once enough simulation observations have been collected, we
look at the cumulative best-so-far across the most recent batches; if it
hasn't improved by at least ``early_stop_min_relative_improvement`` over
``early_stop_batch_window`` batches, we stop.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import qmc

from optimization.common import campaign_dir, ensure_dir, load_config, read_json, write_json
from optimization.loss import compute_loss
from optimization.monitor import refresh_monitor_outputs
from optimization.preprocess import build_fixed10_table, load_sequence_table
from optimization.resident_worker import ensure_worker_pool_submitted
from optimization.rg_parser import parse_rg_file
from optimization.sampling import labels_for_candidate
from optimization.slurm_driver import (
    candidate_name,
    candidate_status,
    prepare_candidate,
    submit_candidate,
)


def ax_available() -> bool:
    return importlib.util.find_spec("ax") is not None


# ---------------------------------------------------------------------------
# Proposal: Sobol for the first ``n_initial_sobol`` trials, then Ax/BoTorch.
# ---------------------------------------------------------------------------


def vector_from_params(config: dict[str, Any], params: dict[str, float]) -> list[float]:
    return [float(params[aa]) for aa in config["amino_acids"]]


def params_from_vector(config: dict[str, Any], vector: list[float] | np.ndarray) -> dict[str, float]:
    lower = float(config["bo"]["lower"])
    upper = float(config["bo"]["upper"])
    vector = np.clip(np.asarray(vector, dtype=float), lower, upper)
    return {aa: float(value) for aa, value in zip(config["amino_acids"], vector)}


def distance_to_existing(vector: np.ndarray, existing: list[list[float]]) -> float:
    if not existing:
        return math.inf
    arr = np.asarray(existing, dtype=float)
    return float(np.linalg.norm(arr - vector[None, :], axis=1).min())


def propose_sobol(config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    """Pick a Sobol candidate maximally far from existing observations."""
    dim = len(config["amino_acids"])
    seed = int(config["bo"].get("sobol_seed", 0)) + int(state.get("next_candidate_index", 0))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    samples = qmc.scale(
        sampler.random_base2(m=7),
        float(config["bo"]["lower"]),
        float(config["bo"]["upper"]),
    )
    existing = [vector_from_params(config, obs["lambda_pw"]) for obs in state.get("observations", [])]
    existing += [vector_from_params(config, c["lambda_pw"]) for c in active_candidates(state)]
    best = max(samples, key=lambda v: distance_to_existing(v, existing))
    return params_from_vector(config, best)


def propose_ax(config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    """Ax (BoTorch) qLogNEI proposal; falls back to Sobol if Ax unavailable
    or we don't yet have enough observations."""
    n_initial = int(config["bo"].get("n_initial_sobol", 8))
    if len(state.get("observations", [])) < n_initial:
        return propose_sobol(config, state)
    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
    except Exception as exc:  # noqa: BLE001
        print(f"[propose] Ax unavailable, falling back to Sobol: {exc}", flush=True)
        return propose_sobol(config, state)

    lower = float(config["bo"]["lower"])
    upper = float(config["bo"]["upper"])
    ax_client = AxClient(
        random_seed=int(config["bo"].get("random_seed", 0)),
        verbose_logging=False,
    )
    ax_client.create_experiment(
        name=config["campaign_name"],
        parameters=[
            {"name": aa, "type": "range", "bounds": [lower, upper], "value_type": "float"}
            for aa in config["amino_acids"]
        ],
        objectives={"loss": ObjectiveProperties(minimize=True)},
    )
    for obs in sorted(state.get("observations", []), key=lambda o: int(o["candidate_index"])):
        params = {aa: float(obs["lambda_pw"][aa]) for aa in config["amino_acids"]}
        _, trial_index = ax_client.attach_trial(params)
        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (float(obs["loss"]), 0.0)})
    for cand in active_candidates(state):
        ax_client.attach_trial({aa: float(cand["lambda_pw"][aa]) for aa in config["amino_acids"]})
    params, _ = ax_client.get_next_trial()
    return {aa: float(params[aa]) for aa in config["amino_acids"]}


def propose_candidate(config: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Returns ``(params, source)`` where ``source`` is ``"sobol"`` for the
    first ``n_initial_sobol`` candidates and ``"botorch"`` afterwards."""
    n_initial = int(config["bo"].get("n_initial_sobol", 8))
    if len(state.get("observations", [])) < n_initial or not ax_available():
        return propose_sobol(config, state), "sobol"
    return propose_ax(config, state), "botorch"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def active_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in state.get("candidates", {}).values() if c.get("status") == "RUNNING"]


def initialize_state(config: dict[str, Any], fixed_labels: list[str]) -> dict[str, Any]:
    root = ensure_dir(campaign_dir(config))
    state_path = root / "state.json"
    state = read_json(state_path, None)
    if state is not None:
        return state
    state = {
        "next_candidate_index": 0,
        "next_batch_index": 0,
        "observations": [],
        "candidates": {},
        "fixed_labels": fixed_labels,
    }
    write_json(state_path, state)
    return state


def save_state(config: dict[str, Any], state: dict[str, Any]) -> None:
    write_json(campaign_dir(config) / "state.json", state)


# ---------------------------------------------------------------------------
# Refresh loop: poll Slurm, finalise completed candidates.
# ---------------------------------------------------------------------------


def candidate_completed(statuses: dict[str, str]) -> bool:
    return bool(statuses) and all(s == "COMPLETED" for s in statuses.values())


def candidate_failed(statuses: dict[str, str]) -> bool:
    failed = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    return any(s in failed for s in statuses.values())


def collect_rg_results(candidate_dir: Path, labels: list[str], burnin_ns: float) -> pd.DataFrame:
    rows = []
    for label in labels:
        run_dir = candidate_dir / "runfile" / label
        rg_path = run_dir / "rg_data.txt"
        if not rg_path.exists():
            raise FileNotFoundError(f"Missing rg_data.txt for {label}: {rg_path}")
        stats = parse_rg_file(rg_path, burnin_fraction=0.5, burnin_ns=burnin_ns)
        stats["label"] = label
        rows.append(stats)
    return pd.DataFrame(rows)


def refresh_running(config: dict[str, Any], state: dict[str, Any], sequence_table: pd.DataFrame) -> None:
    burnin_ns = float(config["objective"].get("burnin_ns", 5.0))
    fixed_labels = list(state["fixed_labels"])
    for key, item in list(state["candidates"].items()):
        if item.get("status") in {"COMPLETED", "FAILED"}:
            continue
        candidate_dir = Path(item["candidate_dir"])
        statuses = candidate_status(candidate_dir)
        item["job_status"] = statuses
        write_json(candidate_dir / "job_status.json", statuses)
        if candidate_completed(statuses):
            rg_df = collect_rg_results(candidate_dir, item["labels"], burnin_ns)
            loss, metrics, _ = compute_loss(config, rg_df, sequence_table, fixed_labels)
            item["status"] = "COMPLETED"
            item["loss"] = float(loss)
            item.update({k: v for k, v in metrics.items() if k != "loss"})
            state["observations"].append(
                {
                    "candidate_index": item["candidate_index"],
                    "trial_key": key,
                    "source": item.get("source", "unknown"),
                    "status": "COMPLETED",
                    "loss": float(loss),
                    "fixed_mae_nm": metrics.get("fixed_mae_nm"),
                    "random_mae_nm": metrics.get("random_mae_nm"),
                    "labels": item["labels"],
                    "lambda_pw": item["lambda_pw"],
                }
            )
            print(f"[complete] {key} loss={loss:.6f} (proposed by {item.get('source')})", flush=True)
        elif candidate_failed(statuses):
            item["status"] = "FAILED"
            print(f"[failed] {key}: {statuses}", flush=True)


# ---------------------------------------------------------------------------
# Submit fresh candidates: stratified-random labels + Sobol/Ax proposal.
# ---------------------------------------------------------------------------


def submit_until_full(
    config: dict[str, Any],
    state: dict[str, Any],
    sequence_table: pd.DataFrame,
    dry_run: bool,
) -> None:
    fixed_labels = list(state["fixed_labels"])
    batch_size = int(config["bo"].get("batch_size", 4))
    active_limit = int(config["bo"].get("active_candidate_limit", batch_size))
    max_trials = config["bo"].get("max_trials")
    n_active = sum(1 for c in state["candidates"].values() if c.get("status") == "RUNNING")
    while n_active < active_limit:
        if max_trials is not None and state["next_candidate_index"] >= int(max_trials):
            return
        batch_index = int(state["next_batch_index"])
        offset = state["next_candidate_index"] % batch_size
        params, source = propose_candidate(config, state)
        labels = labels_for_candidate(config, sequence_table, fixed_labels, batch_index, offset)
        index = int(state["next_candidate_index"])
        # The driver uses ``selected_sequences`` from the config to know which
        # runfiles to stage; for stratified BO we override per candidate.
        effective_config = json.loads(json.dumps(config))
        effective_config["selected_sequences"] = labels
        if dry_run:
            effective_config["slurm"]["submit"] = False
        candidate_dir = prepare_candidate(effective_config, index, params)
        jobs = submit_candidate(effective_config, candidate_dir)
        key = candidate_name(index)
        state["candidates"][key] = {
            "candidate_index": index,
            "candidate_dir": str(candidate_dir),
            "lambda_pw": params,
            "labels": labels,
            "jobs": jobs,
            "status": "RUNNING" if not dry_run else "DRY_RUN",
            "source": source,
        }
        state["next_candidate_index"] = index + 1
        if offset + 1 >= batch_size:
            state["next_batch_index"] = batch_index + 1
        n_active += 1
        print(f"[submit] {key} source={source} labels={len(labels)}", flush=True)
        if dry_run:
            return


# ---------------------------------------------------------------------------
# Persist + plot.
# ---------------------------------------------------------------------------


def write_observation_table(config: dict[str, Any], state: dict[str, Any]) -> None:
    rows = []
    for obs in state["observations"]:
        rows.append(
            {
                "candidate_index": obs["candidate_index"],
                "trial_key": obs["trial_key"],
                "source": obs.get("source"),
                "status": obs["status"],
                "loss": obs["loss"],
                "fixed_mae_nm": obs.get("fixed_mae_nm"),
                "random_mae_nm": obs.get("random_mae_nm"),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(campaign_dir(config) / "observations.csv", index=False)


def cancel_resident_worker_pool(config: dict[str, Any]) -> list[str]:
    queue_name = config.get("resident_worker", {}).get("queue_dir", "resident_queue")
    pool_path = campaign_dir(config) / queue_name / "worker_pool.json"
    pool = read_json(pool_path, {"jobs": []})
    job_ids = [str(item.get("job_id")) for item in pool.get("jobs", []) if item.get("job_id")]
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)
    return job_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async BO controller with stratified-random subsampling."
    )
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    os.environ["BO_CONFIG_PATH"] = str(Path(args.config).resolve())
    config = load_config(args.config)
    if args.dry_run:
        config["slurm"]["submit"] = False

    sequence_table = load_sequence_table(config)
    fixed10_path = config["paths"].get("fixed10_table") or ""
    if fixed10_path and Path(fixed10_path).exists():
        fixed_labels = pd.read_csv(fixed10_path)["label"].astype(str).tolist()
    else:
        # Auto-generate the AA-balanced anchor on first launch.
        fixed_df = build_fixed10_table(config, fixed10_path or None)
        fixed_labels = fixed_df["label"].astype(str).tolist()

    ensure_dir(campaign_dir(config))
    state = initialize_state(config, fixed_labels)
    print(f"[bo] backend={'Ax' if ax_available() else 'SobolFallback'} fixed10={len(fixed_labels)}")

    while True:
        if not args.dry_run:
            refresh_running(config, state, sequence_table)
        write_observation_table(config, state)
        refresh_monitor_outputs(config, state)
        save_state(config, state)
        n_completed = sum(1 for o in state["observations"] if o["status"] == "COMPLETED")
        max_trials = config["bo"].get("max_trials")
        if max_trials is not None and n_completed >= int(max_trials):
            cancel_resident_worker_pool(config)
            print(f"[done] reached max_trials={max_trials}", flush=True)
            break
        submit_until_full(config, state, sequence_table, args.dry_run)
        if not args.dry_run and str(config["slurm"].get("launcher", "direct")).lower() == "resident":
            ensure_worker_pool_submitted(config)
        save_state(config, state)
        if args.once or args.dry_run:
            break
        time.sleep(int(config["slurm"].get("poll_seconds", 60)))


if __name__ == "__main__":
    main()
