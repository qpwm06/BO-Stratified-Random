"""Persistent ('resident') Slurm worker pool that consumes a local task queue.

The BO controller produces work much faster than Slurm allocates new GPUs,
so each candidate sequence would otherwise spend most of its time waiting
in the cluster queue. Instead, the controller submits a fixed-size pool of
long-walltime worker jobs once. Each worker:

  * Takes one PENDING task from ``resident_queue/tasks/`` via an atomic
    ``mkdir`` lock under ``resident_queue/locks/``.
  * Runs the simulation in the task's ``run_dir`` with the configured
    ``simulation_python`` interpreter.
  * On finish, marks the task COMPLETED/FAILED and waits for the next one.
  * Heartbeats every poll cycle so a controller-side watchdog can re-queue
    tasks abandoned by a crashed worker.

A short ``stop_launch_within_hours`` window prevents starting new tasks
just before the worker's walltime expires; if ``requeue_on_stop`` is set
the worker submits its own replacement before exiting.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization.common import campaign_dir, ensure_dir, load_config, read_json, write_json


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def queue_root(config: dict[str, Any]) -> Path:
    queue_name = config.get("resident_worker", {}).get("queue_dir", "resident_queue")
    return ensure_dir(campaign_dir(config) / queue_name)


def task_path(config: dict[str, Any], task_id: str) -> Path:
    return queue_root(config) / "tasks" / f"{task_id}.json"


def read_task(config: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    return read_json(task_path(config, task_id), None)


def write_task(config: dict[str, Any], task: dict[str, Any]) -> None:
    write_json(task_path(config, task["task_id"]), task)


def mark_task_failed(config: dict[str, Any], task: dict[str, Any], reason: str, returncode: int = -1) -> None:
    task.update(
        {
            "status": "FAILED",
            "returncode": returncode,
            "failure_reason": reason,
            "finished_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    write_task(config, task)
    release_task_lock(config, task["task_id"])


def requeue_task(config: dict[str, Any], task: dict[str, Any], reason: str) -> None:
    """Reset a task to PENDING with one extra attempt counter."""
    attempts = int(task.get("attempts", 0)) + 1
    history = list(task.get("history", []))
    history.append(
        {
            "status": task.get("status"),
            "worker_job_id": task.get("worker_job_id"),
            "worker_id": task.get("worker_id"),
            "reason": reason,
            "time": utc_now(),
        }
    )
    task.update(
        {
            "status": "PENDING",
            "attempts": attempts,
            "history": history,
            "returncode": None,
            "failure_reason": None,
            "worker_job_id": None,
            "worker_id": None,
            "started_at": None,
            "finished_at": None,
            "heartbeat_at": None,
            "updated_at": utc_now(),
        }
    )
    write_task(config, task)
    release_task_lock(config, task["task_id"])


def watchdog_file_exists(config: dict[str, Any], run_dir: Path) -> bool:
    """Returns True once the simulation has actually started writing output."""
    names = config.get("resident_worker", {}).get("startup_watchdog_files", ["thermo_md_log.txt"])
    return any((run_dir / str(name)).exists() for name in names)


def startup_watchdog_expired(config: dict[str, Any], task: dict[str, Any]) -> bool:
    """A worker that hasn't produced its first log line within N minutes is
    treated as stuck (e.g. GPU init failure), and its task is re-queued."""
    minutes = float(config.get("resident_worker", {}).get("startup_watchdog_minutes", 5))
    started = parse_utc(task.get("started_at"))
    if started is None:
        return False
    if time.time() - started < minutes * 60:
        return False
    return not watchdog_file_exists(config, Path(task["run_dir"]))


def heartbeat_task(config: dict[str, Any], task: dict[str, Any]) -> None:
    current = read_task(config, task["task_id"])
    if not current:
        return
    current["heartbeat_at"] = utc_now()
    current["updated_at"] = utc_now()
    write_task(config, current)


def cleanup_stale_tasks(config: dict[str, Any]) -> int:
    """Re-flag tasks whose worker job died without updating its task file."""
    stale_minutes = float(config.get("resident_worker", {}).get("stale_task_minutes", 20))
    threshold = time.time() - stale_minutes * 60
    tasks_dir = queue_root(config) / "tasks"
    if not tasks_dir.exists():
        return 0
    cleaned = 0
    for path in sorted(tasks_dir.glob("*.json")):
        task = read_json(path, None)
        if not task or task.get("status") != "RUNNING":
            continue
        stamp = parse_utc(task.get("heartbeat_at") or task.get("updated_at"))
        if stamp is not None and stamp >= threshold:
            continue
        worker_job_id = str(task.get("worker_job_id") or "")
        worker_state = query_slurm_job_state(worker_job_id) if worker_job_id else "UNKNOWN"
        if worker_state not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            mark_task_failed(config, task, f"stale worker state={worker_state}")
            cleaned += 1
    return cleaned


def claim_task(config: dict[str, Any], worker_id: str) -> dict[str, Any] | None:
    """Atomic claim: ``mkdir`` on the lock directory wins exactly one worker."""
    tasks_dir = queue_root(config) / "tasks"
    locks_dir = ensure_dir(queue_root(config) / "locks")
    if not tasks_dir.exists():
        return None
    for path in sorted(tasks_dir.glob("*.json")):
        task = read_json(path, None)
        if not task or task.get("status") != "PENDING":
            continue
        lock_path = locks_dir / f"{path.stem}.lock"
        try:
            os.mkdir(lock_path)
        except FileExistsError:
            continue
        task.update({"status": "CLAIMED", "worker_id": worker_id, "claimed_at": utc_now(), "updated_at": utc_now()})
        write_json(path, task)
        return task
    return None


def release_task_lock(config: dict[str, Any], task_id: str) -> None:
    lock_path = queue_root(config) / "locks" / f"{task_id}.lock"
    try:
        lock_path.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


def enqueue_task(config: dict[str, Any], candidate_dir: Path, label: str) -> str:
    candidate_name = candidate_dir.name
    task_id = f"{candidate_name}__{label}"
    run_dir = candidate_dir / "runfile" / label
    task = read_task(config, task_id)
    if task and task.get("status") not in TERMINAL_STATES:
        return task_id
    write_task(
        config,
        {
            "task_id": task_id,
            "candidate": candidate_name,
            "label": label,
            "run_dir": str(run_dir.resolve()),
            "status": "PENDING",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "returncode": None,
            "worker_job_id": None,
        },
    )
    return task_id


def resident_job_id(task_id: str) -> str:
    return f"RESIDENT:{task_id}"


def resident_task_id(job_id: str) -> str | None:
    prefix = "RESIDENT:"
    if job_id.startswith(prefix):
        return job_id[len(prefix) :]
    return None


def query_resident_state(config: dict[str, Any], job_id: str) -> str:
    task_id = resident_task_id(job_id)
    if task_id is None:
        return "UNKNOWN"
    task = read_task(config, task_id)
    if not task:
        return "UNKNOWN"
    return str(task.get("status", "UNKNOWN"))


def pending_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks_dir = queue_root(config) / "tasks"
    if not tasks_dir.exists():
        return []
    tasks = []
    for path in sorted(tasks_dir.glob("*.json")):
        task = read_json(path, None)
        if task and task.get("status") == "PENDING":
            tasks.append(task)
    return tasks


def seconds_left(start_time: float, walltime_seconds: int) -> float:
    return walltime_seconds - (time.time() - start_time)


def parse_slurm_time(value: str) -> int:
    text = str(value)
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = parts[0], 0, 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def start_task(config: dict[str, Any], task: dict[str, Any], worker_id: str) -> subprocess.Popen:
    run_dir = Path(task["run_dir"])
    log_path = run_dir / "resident-worker.log"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OMP_NUM_THREADS"] = "1"
    task.update(
        {
            "status": "RUNNING",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "worker_id": worker_id,
            "worker_job_id": os.environ.get("SLURM_JOB_ID"),
            "log_path": str(log_path),
        }
    )
    write_task(config, task)
    log_handle = log_path.open("ab")
    command = [str(config["paths"]["simulation_python"]), "production.py"]
    process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    process._resident_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def finish_task(config: dict[str, Any], task: dict[str, Any], returncode: int) -> None:
    status = "COMPLETED" if returncode == 0 else "FAILED"
    task.update({"status": status, "returncode": returncode, "finished_at": utc_now(), "updated_at": utc_now()})
    write_task(config, task)
    release_task_lock(config, task["task_id"])


def record_worker_job(config: dict[str, Any], job_id: str, *, replacement_for: str | None = None) -> None:
    pool_path = queue_root(config) / "worker_pool.json"
    pool = read_json(pool_path, {"jobs": []})
    jobs = [item for item in pool.get("jobs", []) if str(item.get("job_id", "")) != str(job_id)]
    item = {"job_id": str(job_id), "submitted_at": utc_now(), "last_state": "SUBMITTED"}
    if replacement_for:
        item["replacement_for"] = replacement_for
    jobs.append(item)
    pool["jobs"] = jobs
    pool["updated_at"] = utc_now()
    write_json(pool_path, pool)


def write_worker_slurm(config: dict[str, Any]) -> Path:
    worker = config["resident_worker"]
    root = queue_root(config)
    script = root / "resident_worker.slurm"
    log_dir = ensure_dir(root / "logs")
    config_path = Path(config["_config_path"]).resolve()
    module_block = list(worker.get("module_block", config["slurm"].get("module_block", [])))
    lines = [
        "#!/bin/bash\n",
        "#SBATCH --job-name=bo_resident_worker\n",
        f"#SBATCH --time={worker['time']}\n",
        f"#SBATCH --mem={worker['mem']}\n",
        f"#SBATCH --output={log_dir}/resident-%j-%N.out\n",
        f"#SBATCH --gres={worker['gres']}\n",
        f"#SBATCH --partition={worker['partition']}\n",
        f"#SBATCH --nodes={worker['nodes']}\n",
        f"#SBATCH --ntasks-per-node={worker['ntasks_per_node']}\n",
        "#SBATCH --mail-type=NONE\n",
        "\n",
        "# TODO: load whichever modules / activate whichever env your\n",
        "# cluster needs. Populate ``slurm.module_block`` (or\n",
        "# ``resident_worker.module_block``) in the config to emit lines.\n",
    ]
    for module_line in module_block:
        lines.append(f"{module_line}\n")
    lines.extend(
        [
            f"cd {Path.cwd().resolve()}\n",
            f"python -m optimization.resident_worker run --config {config_path}\n",
        ]
    )
    script.write_text("".join(lines), encoding="utf-8")
    return script


def submit_worker(config: dict[str, Any]) -> str:
    script = write_worker_slurm(config)
    result = subprocess.run(["sbatch", str(script)], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    job_id = result.stdout.strip().split()[-1]
    write_json(queue_root(config) / "worker_job.json", {"job_id": job_id, "submitted_at": utc_now(), "script": str(script)})
    record_worker_job(config, job_id)
    return job_id


def query_slurm_job_state(job_id: str) -> str:
    result = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"], text=True, capture_output=True, check=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    result = subprocess.run(["sacct", "-j", job_id, "--format=State", "--noheader", "-P"], text=True, capture_output=True, check=False)
    if result.returncode == 0:
        states = [line.split("|")[0].strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        if states:
            return states[0]
    return "UNKNOWN"


def ensure_worker_pool_submitted(config: dict[str, Any]) -> list[str]:
    """Maintain a target number of active worker jobs; submit replacements
    until the count is met. Called by the BO controller every loop."""
    if not config.get("resident_worker", {}).get("enabled", False):
        return []
    worker = config["resident_worker"]
    cleanup_stale_tasks(config)
    target = int(worker.get("num_workers", 1))
    pool_path = queue_root(config) / "worker_pool.json"
    pool = read_json(pool_path, {"jobs": []})
    active_jobs = []
    for item in pool.get("jobs", []):
        job_id = str(item.get("job_id", ""))
        if not job_id:
            continue
        state = query_slurm_job_state(job_id)
        item["last_state"] = state
        item["checked_at"] = utc_now()
        if state in {"PENDING", "RUNNING", "CONFIGURING"}:
            active_jobs.append(item)

    while len(active_jobs) < target:
        job_id = submit_worker(config)
        active_jobs.append({"job_id": job_id, "submitted_at": utc_now(), "last_state": "SUBMITTED"})
    write_json(pool_path, {"jobs": active_jobs, "target": target, "updated_at": utc_now()})
    return [item["job_id"] for item in active_jobs]


def ensure_worker_submitted(config: dict[str, Any]) -> str | None:
    jobs = ensure_worker_pool_submitted(config)
    return ",".join(jobs) if jobs else None


def run_worker(config: dict[str, Any]) -> None:
    """Main loop of one resident worker process inside its Slurm allocation."""
    worker = config["resident_worker"]
    poll_seconds = int(worker.get("poll_seconds", 30))
    walltime_seconds = parse_slurm_time(worker["time"])
    stop_launch_seconds = int(float(worker.get("stop_launch_within_hours", 5)) * 3600)
    start_time = time.time()
    worker_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:8]}"
    running: tuple[subprocess.Popen, dict[str, Any]] | None = None
    stopping = False
    requeued = False

    def handle_stop(signum: int, frame: Any) -> None:
        nonlocal stopping
        stopping = True
        print(f"[resident] received signal {signum}; stop launching new tasks", flush=True)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    print(f"[resident] start worker_id={worker_id} stop_launch_within={stop_launch_seconds}s", flush=True)

    while True:
        if running is not None:
            process, task = running
            returncode = process.poll()
            if returncode is not None:
                log_handle = getattr(process, "_resident_log_handle", None)
                if log_handle is not None:
                    log_handle.close()
                finish_task(config, task, int(returncode))
                running = None
                print(f"[resident] finish {task['task_id']} rc={returncode}", flush=True)
            elif stopping:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                log_handle = getattr(process, "_resident_log_handle", None)
                if log_handle is not None:
                    log_handle.close()
                mark_task_failed(config, task, "worker received stop signal", returncode=-15)
                running = None
                print(f"[resident] stopped {task['task_id']} after signal", flush=True)
            elif startup_watchdog_expired(config, task):
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                log_handle = getattr(process, "_resident_log_handle", None)
                if log_handle is not None:
                    log_handle.close()
                requeue_task(config, task, "startup watchdog missing log file")
                running = None
                print(f"[resident] requeue {task['task_id']} missing log; worker exits", flush=True)
                break
            else:
                heartbeat_task(config, task)

        left = seconds_left(start_time, walltime_seconds)
        can_launch = not stopping and left > stop_launch_seconds
        if not can_launch and not stopping and not requeued and bool(worker.get("requeue_on_stop", True)):
            requeued = True
            try:
                job_id = submit_worker(config)
                record_worker_job(config, job_id, replacement_for=os.environ.get("SLURM_JOB_ID"))
                print(f"[resident] submitted replacement worker job={job_id}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[resident] replacement submit failed: {exc}", flush=True)
        if can_launch and running is None:
            task = claim_task(config, worker_id)
            if task:
                process = start_task(config, task, worker_id)
                running = (process, task)
                print(f"[resident] start {task['task_id']}", flush=True)

        if running is None and not can_launch:
            print("[resident] stop window reached and no running tasks; exit for requeue", flush=True)
            break
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Slurm worker pool consuming a local task queue.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("write-slurm", "submit-worker", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="config.example.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "write-slurm":
        print(write_worker_slurm(config))
    elif args.command == "submit-worker":
        print(submit_worker(config))
    elif args.command == "run":
        run_worker(config)


if __name__ == "__main__":
    main()
