from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import subprocess
import time
from typing import Any

import pandas as pd

from optimization.common import campaign_dir, ensure_dir, load_config, write_json
from optimization.loss import write_loss_outputs
from optimization.patch_sim import prepare_sequence_run
from optimization.resident_worker import enqueue_task, query_resident_state, resident_job_id
from optimization.rg_compute import ensure_rg_data
from optimization.rg_parser import parse_candidate

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
SUCCESS_STATES = {"COMPLETED"}

# The resident-worker state lookup needs the active config. ``bo_loop``
# sets this before any state queries; users with a custom entry point can
# override ``BO_CONFIG_PATH`` in the environment.
DEFAULT_CONFIG_PATH = Path(os.environ.get("BO_CONFIG_PATH", "config.example.yaml"))


def candidate_name(index: int) -> str:
    return f"candidate_{index:04d}"


def prepare_candidate(config: dict[str, Any], index: int, parameters: dict[str, float]) -> Path:
    root = ensure_dir(campaign_dir(config) / "candidates")
    name = candidate_name(index)
    candidate_dir = ensure_dir(root / name)
    write_json(candidate_dir / "params.json", {"candidate_index": index, "lambda_pw": parameters})
    for label in config["selected_sequences"]:
        prepare_sequence_run(config, candidate_dir, label, parameters, name)
    return candidate_dir


def submit_job(run_dir: Path) -> str:
    result = subprocess.run(["sbatch", "job.slurm"], cwd=run_dir, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip().split()[-1]


def run_locally(config: dict[str, Any], run_dir: Path) -> str:
    """Run ``simulation_python`` synchronously inside ``run_dir``.

    Used by the ``local`` launcher; convenient for smoke tests with the
    bundled fake runner, where there is no real Slurm scheduler around.
    """
    interpreter = str(config["paths"]["simulation_python"])
    result = subprocess.run([interpreter, "production.py"], cwd=run_dir, check=False)
    return "COMPLETED" if result.returncode == 0 else "FAILED"


def submit_candidate(config: dict[str, Any], candidate_dir: Path) -> dict[str, str]:
    """Hand off every sequence in this candidate.

    Three launch modes are supported:

      * ``direct``: each sequence becomes its own ``sbatch`` submission.
      * ``resident``: the sequence is appended to the local task queue
        consumed by the persistent worker pool (see ``resident_worker``).
      * ``local``: run the simulation synchronously in-process. Intended
        for smoke tests with the fake runner; not for real workloads.
    """
    jobs: dict[str, str] = {}
    launcher = str(config["slurm"].get("launcher", "direct")).lower()
    for label in config["selected_sequences"]:
        run_dir = candidate_dir / "runfile" / label
        if launcher == "resident":
            task_id = enqueue_task(config, candidate_dir, label)
            jobs[label] = resident_job_id(task_id)
        elif launcher == "local":
            jobs[label] = run_locally(config, run_dir)
        elif config["slurm"].get("submit", True):
            jobs[label] = submit_job(run_dir)
        else:
            jobs[label] = "DRY_RUN"
    write_json(candidate_dir / "jobs.json", jobs)
    return jobs


def query_job_state(job_id: str) -> str:
    if job_id == "DRY_RUN":
        return "DRY_RUN"
    if job_id in TERMINAL_STATES:
        return job_id
    if job_id.startswith("RESIDENT:"):
        config = load_config(DEFAULT_CONFIG_PATH)
        return query_resident_state(config, job_id)
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        states = [line.split("|")[0].strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        terminal = [state for state in states if state in TERMINAL_STATES]
        if terminal:
            if any(state in SUCCESS_STATES for state in terminal):
                return "COMPLETED"
            return terminal[0]
    result = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"], text=True, capture_output=True, check=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    return "UNKNOWN"


def candidate_status(candidate_dir: Path) -> dict[str, str]:
    jobs_path = candidate_dir / "jobs.json"
    if not jobs_path.exists():
        return {}
    import json

    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    return {label: query_job_state(job_id) for label, job_id in jobs.items()}


def finalize_candidate(config: dict[str, Any], candidate_dir: Path) -> float:
    for label in config["selected_sequences"]:
        ensure_rg_data(candidate_dir / "runfile" / label)
    objective = config["objective"]
    rg_df = parse_candidate(
        candidate_dir,
        list(config["selected_sequences"]),
        float(objective["burnin_fraction"]),
        objective.get("burnin_ns"),
    )
    rg_df.to_csv(candidate_dir / "parsed_rg.csv", index=False)
    return write_loss_outputs(config, candidate_dir, rg_df)


def wait_candidate(config: dict[str, Any], candidate_dir: Path) -> str:
    poll_seconds = int(config["slurm"].get("poll_seconds", 300))
    while True:
        statuses = candidate_status(candidate_dir)
        write_json(candidate_dir / "job_status.json", statuses)
        if statuses and all(state == "COMPLETED" for state in statuses.values()):
            return "COMPLETED"
        failed = {label: state for label, state in statuses.items() if state in TERMINAL_STATES - SUCCESS_STATES}
        if failed:
            write_json(candidate_dir / "failed_jobs.json", failed)
            return "FAILED"
        print(f"[wait] {candidate_dir.name}: {statuses}")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare / submit / poll a single BO candidate.")
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--params-csv", required=True, help="Two columns: aa, value")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    params = pd.read_csv(args.params_csv)
    parameters = dict(zip(params["aa"], params["value"]))
    candidate_dir = prepare_candidate(config, args.candidate_index, parameters)
    print(f"[prepare] {candidate_dir}")
    if args.submit:
        jobs = submit_candidate(config, candidate_dir)
        print(jobs)


if __name__ == "__main__":
    main()
