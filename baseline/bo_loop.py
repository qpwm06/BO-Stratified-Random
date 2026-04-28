"""Asynchronous Bayesian-optimization controller.

Outer loop:

  1. Refresh status of every running candidate (via Slurm or the resident
     queue), finalising any whose sequence jobs have all completed.
  2. Persist ``state.json``, ``observations.csv``, ``status.csv``,
     ``performance.png``.
  3. Top up the running set: while ``len(running) < max_parallel_candidates``
     and ``completed_sim < max_trials``, propose a new candidate (Ax if
     installed, Sobol fallback otherwise), stage its runfiles, and submit.
  4. If the launcher is ``resident``, ensure the worker pool stays at its
     target size (submitting replacement workers as needed).
  5. Sleep ``slurm.poll_seconds`` and repeat.

Early-stop: when no candidates are running and the last N batches show no
improvement in best-so-far loss, the controller cancels the worker pool
and exits.
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
from optimization.monitor import refresh_monitor_outputs, resident_panel_counts as monitor_resident_panel_counts
from baseline.preprocess import build_initial_data, discover_rg_summaries
from optimization.resident_worker import ensure_worker_pool_submitted
from optimization.slurm_driver import (
    candidate_name,
    candidate_status,
    finalize_candidate,
    prepare_candidate,
    submit_candidate,
)


def resident_panel_counts(config: dict[str, Any]) -> tuple[int, int]:
    counts = monitor_resident_panel_counts(config)
    return int(counts.get("updated_jobs", 0)), int(counts.get("thermo_md_jobs", 0))


def ax_available() -> bool:
    return importlib.util.find_spec("ax") is not None


def load_default_parameters(config: dict[str, Any]) -> dict[str, float]:
    """Optional starting point for warm-start synthetic observations.

    The original pipeline parsed defaults out of an engine-specific ``production.py``;
    in the public release the user either provides them via
    ``parameter_space.defaults_csv`` (two columns ``aa,value``) or accepts
    the midpoint of the bounds.
    """
    defaults_csv = config["parameter_space"].get("defaults_csv")
    if defaults_csv:
        df = pd.read_csv(defaults_csv)
        return {str(row.aa): float(row.value) for row in df.itertuples(index=False)}
    midpoint = 0.5 * (
        float(config["parameter_space"]["lower"]) + float(config["parameter_space"]["upper"])
    )
    return {aa: midpoint for aa in config["amino_acids"]}


def alpha_warm_start_params(default_parameters: dict[str, float], alpha_weight: float) -> dict[str, float]:
    """Synthesise a parameter vector for an upstream sweep that only logged a
    single mixing scalar. The resulting vector has elevated noise so the BO
    treats it as a soft prior rather than a hard observation."""
    return {
        aa: min(1.0, max(0.0, value + 0.2 * (alpha_weight - 0.2)))
        for aa, value in default_parameters.items()
    }


def initialize_state(config: dict[str, Any]) -> dict[str, Any]:
    root = ensure_dir(campaign_dir(config))
    state_path = root / "state.json"
    state = read_json(state_path, None)
    if state is not None:
        return state

    default_parameters = load_default_parameters(config)
    observations: list[dict[str, Any]] = []
    if config.get("paths", {}).get("data_root"):
        try:
            rg_df = discover_rg_summaries(config)
            initial_df = build_initial_data(config, rg_df)
            for row in initial_df.itertuples(index=False):
                params = alpha_warm_start_params(default_parameters, 0.0)
                observations.append(
                    {
                        "candidate_index": None,
                        "trial_key": row.trial_key,
                        "source": row.source,
                        "status": "COMPLETED",
                        "loss": float(row.loss),
                        "noise_sd": float(row.noise_sd),
                        "lambda_pw": params,
                    }
                )
        except FileNotFoundError:
            # No warm-start data available; start cold.
            pass

    state = {
        "next_candidate_index": 0,
        "observations": observations,
        "running": {},
        "failed": [],
        "default_parameters": default_parameters,
    }
    write_json(state_path, state)
    return state


def save_state(config: dict[str, Any], state: dict[str, Any]) -> None:
    write_json(campaign_dir(config) / "state.json", state)


def vector_from_params(config: dict[str, Any], params: dict[str, float]) -> list[float]:
    return [float(params[aa]) for aa in config["amino_acids"]]


def params_from_vector(config: dict[str, Any], vector: list[float]) -> dict[str, float]:
    lower = float(config["parameter_space"]["lower"])
    upper = float(config["parameter_space"]["upper"])
    return {aa: float(min(upper, max(lower, value))) for aa, value in zip(config["amino_acids"], vector)}


def distance_to_existing(vector: np.ndarray, existing: list[list[float]]) -> float:
    if not existing:
        return math.inf
    arr = np.asarray(existing, dtype=float)
    return float(np.linalg.norm(arr - vector[None, :], axis=1).min())


def propose_with_sobol(config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    """Sobol fallback: pick the candidate maximally far from existing ones."""
    dim = len(config["amino_acids"])
    lower = float(config["parameter_space"]["lower"])
    upper = float(config["parameter_space"]["upper"])
    seed = int(config["bo"]["random_seed"]) + int(state["next_candidate_index"])
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    samples = sampler.random_base2(m=7)
    existing = [vector_from_params(config, obs["lambda_pw"]) for obs in state["observations"]]
    existing += [vector_from_params(config, item["lambda_pw"]) for item in state["running"].values()]
    scaled = qmc.scale(samples, lower, upper)
    best = max(scaled, key=lambda vector: distance_to_existing(vector, existing))
    return params_from_vector(config, best.tolist())


def propose_with_ax(config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    """Ax (BoTorch Modular) acquisition: defaults to qLogNoisyEI for a
    minimisation objective with noisy warm-start observations."""
    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
    except Exception:
        return propose_with_sobol(config, state)

    lower = float(config["parameter_space"]["lower"])
    upper = float(config["parameter_space"]["upper"])
    ax_client = AxClient(random_seed=int(config["bo"]["random_seed"]), verbose_logging=False)
    parameters = [
        {"name": aa, "type": "range", "bounds": [lower, upper], "value_type": "float"}
        for aa in config["amino_acids"]
    ]
    ax_client.create_experiment(
        name=config["campaign_name"],
        parameters=parameters,
        objectives={"loss": ObjectiveProperties(minimize=True)},
    )
    for obs in state["observations"]:
        params = {aa: float(obs["lambda_pw"][aa]) for aa in config["amino_acids"]}
        _, trial_index = ax_client.attach_trial(params)
        sem = float(obs.get("noise_sd", 0.0)) if obs.get("source", "").startswith("warm_start") else 0.0
        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (float(obs["loss"]), sem)})
    for item in state["running"].values():
        ax_client.attach_trial({aa: float(item["lambda_pw"][aa]) for aa in config["amino_acids"]})
    params, _ = ax_client.get_next_trial()
    return {aa: float(params[aa]) for aa in config["amino_acids"]}


def propose_candidate(config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    if ax_available():
        return propose_with_ax(config, state)
    return propose_with_sobol(config, state)


def running_completed(statuses: dict[str, str]) -> bool:
    return bool(statuses) and all(state == "COMPLETED" for state in statuses.values())


def running_failed(statuses: dict[str, str]) -> bool:
    failed_states = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    return any(state in failed_states for state in statuses.values())


def simulation_observations(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [obs for obs in state.get("observations", []) if obs.get("source") == "simulation"]


def batch_best_losses(state: dict[str, Any], batch_size: int) -> list[float]:
    observations = sorted(simulation_observations(state), key=lambda obs: int(obs["candidate_index"]))
    losses = []
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        if len(batch) == batch_size:
            losses.append(min(float(obs["loss"]) for obs in batch))
    return losses


def should_early_stop(config: dict[str, Any], state: dict[str, Any]) -> bool:
    patience = int(config["bo"].get("early_stop_batches_without_improvement", 0))
    if patience <= 0 or state.get("running"):
        return False
    batch_size = int(config["bo"].get("max_parallel_candidates", 1))
    losses = batch_best_losses(state, batch_size)
    if len(losses) <= patience:
        return False
    previous_best = min(losses[:-patience])
    recent_best = min(losses[-patience:])
    return recent_best >= previous_best


def cancel_resident_worker_pool(config: dict[str, Any]) -> list[str]:
    queue_name = config.get("resident_worker", {}).get("queue_dir", "resident_queue")
    pool_path = campaign_dir(config) / queue_name / "worker_pool.json"
    pool = read_json(pool_path, {"jobs": []})
    job_ids = [str(item.get("job_id")) for item in pool.get("jobs", []) if item.get("job_id")]
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)
    return job_ids


def refresh_running(config: dict[str, Any], state: dict[str, Any]) -> None:
    for key, item in list(state["running"].items()):
        candidate_dir = Path(item["candidate_dir"])
        statuses = candidate_status(candidate_dir)
        item["job_status"] = statuses
        write_json(candidate_dir / "job_status.json", statuses)
        if running_completed(statuses):
            loss = finalize_candidate(config, candidate_dir)
            state["observations"].append(
                {
                    "candidate_index": item["candidate_index"],
                    "trial_key": key,
                    "source": "simulation",
                    "status": "COMPLETED",
                    "loss": loss,
                    "noise_sd": 0.0,
                    "lambda_pw": item["lambda_pw"],
                }
            )
            del state["running"][key]
            print(f"[complete] {key} loss={loss:.6f}")
            refresh_monitor_outputs(config, state)
        elif running_failed(statuses):
            item["status"] = "FAILED"
            state["failed"].append(item)
            del state["running"][key]
            print(f"[failed] {key}: {statuses}")
            refresh_monitor_outputs(config, state)


def submit_until_full(config: dict[str, Any], state: dict[str, Any], dry_run: bool) -> None:
    max_parallel = int(config["bo"]["max_parallel_candidates"])
    max_trials = int(config["bo"]["max_trials"])
    completed_sim = sum(1 for obs in state["observations"] if obs.get("source") == "simulation")
    submit_mode = str(config["bo"].get("submit_mode", "streaming")).lower()
    if submit_mode == "batch" and state["running"]:
        return
    while len(state["running"]) < max_parallel and completed_sim + len(state["running"]) < max_trials:
        index = int(state["next_candidate_index"])
        params = propose_candidate(config, state)
        candidate_dir = prepare_candidate(config, index, params)
        effective_config = json.loads(json.dumps(config))
        if dry_run:
            effective_config["slurm"]["submit"] = False
        jobs = submit_candidate(effective_config, candidate_dir)
        key = candidate_name(index)
        state["running"][key] = {
            "candidate_index": index,
            "candidate_dir": str(candidate_dir),
            "lambda_pw": params,
            "jobs": jobs,
            "status": "RUNNING" if not dry_run else "DRY_RUN",
        }
        state["next_candidate_index"] = index + 1
        print(f"[submit] {key}: {candidate_dir}", flush=True)
        refresh_monitor_outputs(config, state)
        completed_sim = sum(1 for obs in state["observations"] if obs.get("source") == "simulation")
        if dry_run:
            break


def ensure_resident_worker_if_needed(config: dict[str, Any], state: dict[str, Any], dry_run: bool) -> None:
    """Keep the resident worker pool topped up while there are queued tasks."""
    if dry_run:
        return
    if str(config["slurm"].get("launcher", "direct")).lower() != "resident":
        return
    has_resident_task = any(
        str(job_id).startswith("RESIDENT:")
        for item in state.get("running", {}).values()
        for job_id in item.get("jobs", {}).values()
    )
    if not has_resident_task:
        return
    job_ids = ensure_worker_pool_submitted(config)
    if job_ids:
        updated_jobs, thermo_jobs = resident_panel_counts(config)
        print(
            f"[resident] active worker jobs={len(job_ids)} | "
            f"updated jobs={updated_jobs} | thermo jobs={thermo_jobs}",
            flush=True,
        )


def write_observation_table(config: dict[str, Any], state: dict[str, Any]) -> None:
    rows = []
    for obs in state["observations"]:
        row = {
            "trial_key": obs["trial_key"],
            "source": obs["source"],
            "status": obs["status"],
            "loss": obs["loss"],
        }
        row.update({f"lambda_pw_{aa}": value for aa, value in obs["lambda_pw"].items()})
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(campaign_dir(config) / "observations.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Async Bayesian-optimization controller.")
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Generate candidates without sbatch.")
    parser.add_argument("--max-trials", type=int, default=None, help="Override config's BO trial cap.")
    parser.add_argument("--once", action="store_true", help="Run a single refresh+submit pass and exit.")
    args = parser.parse_args()

    # Make the config path discoverable to slurm_driver.query_job_state.
    os.environ["BO_CONFIG_PATH"] = str(Path(args.config).resolve())
    config = load_config(args.config)
    if args.max_trials is not None:
        config["bo"]["max_trials"] = args.max_trials
    if args.dry_run:
        config["slurm"]["submit"] = False

    ensure_dir(campaign_dir(config))
    state = initialize_state(config)
    print(f"[bo] backend={'Ax' if ax_available() else 'SobolFallback'}")

    while True:
        if not args.dry_run:
            refresh_running(config, state)
        write_observation_table(config, state)
        refresh_monitor_outputs(config, state)
        save_state(config, state)
        completed_sim = sum(1 for obs in state["observations"] if obs.get("source") == "simulation")
        if should_early_stop(config, state):
            cancelled = cancel_resident_worker_pool(config)
            print(
                f"[done] early stop: last {config['bo'].get('early_stop_batches_without_improvement')} "
                f"batches did not improve; cancelled worker jobs={','.join(cancelled)}",
                flush=True,
            )
            break
        if completed_sim >= int(config["bo"]["max_trials"]):
            cancelled = cancel_resident_worker_pool(config)
            print(f"[done] reached max_trials; cancelled worker jobs={','.join(cancelled)}", flush=True)
            break
        submit_until_full(config, state, args.dry_run)
        ensure_resident_worker_if_needed(config, state, args.dry_run)
        write_observation_table(config, state)
        refresh_monitor_outputs(config, state)
        save_state(config, state)
        if args.once or args.dry_run:
            break
        time.sleep(int(config["slurm"].get("poll_seconds", 300)))


if __name__ == "__main__":
    main()
