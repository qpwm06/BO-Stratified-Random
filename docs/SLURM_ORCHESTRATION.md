# Slurm orchestration

## Why a custom orchestrator

A naive workflow submits one Slurm job per simulation and waits. With
batched BO that means thousands of `sbatch` calls per campaign, each
paying the cluster-queue tax (30 s – 30 min depending on load). The
controller spends more time *waiting* than the GPUs spend *working*.

This repo replaces that with a **persistent worker pool** plus a local
on-disk task queue. The BO controller files tasks into a directory and
the workers pull from it. Slurm only sees *N* long-walltime worker
allocations, not thousands of micro-jobs.

Three launchers ship with the repo (see `slurm.launcher` in the
config):

| launcher  | what `submit_candidate` does                                   | when to use            |
|-----------|-----------------------------------------------------------------|------------------------|
| `direct`  | issues one `sbatch` per (candidate × sequence)                 | single-shot, no pool   |
| `resident`| writes JSON tasks to `runs/<campaign>/resident_queue/tasks/`   | the production setting |
| `local`   | runs the simulation synchronously in-process                   | the bundled fake-runner |

## Resident worker pool

`optimization/resident_worker.py` is a long-running worker loop. The
controller starts *N* of them via `sbatch`; each one:

1. Looks at the queue and atomically claims one PENDING task using
   `mkdir(<task_id>.lock)`. The first worker to win the directory
   creation wins the task; the rest move on. No central broker, no
   races.
2. Runs the simulation in the task's `run_dir` with
   `paths.simulation_python` (point this at your engine's Python build
   in production, or at `optimization/fake_run.py` for a smoke test).
3. Marks the task COMPLETED / FAILED in its JSON file.
4. Heartbeats every poll cycle so a controller-side watchdog can
   re-flag tasks abandoned by a crashed worker.

```
┌──────────────────────┐       ┌──────────────────────────────┐
│  bo_loop.py          │       │ resident_queue/tasks/*.json  │
│  (controller)        │──────▶│ resident_queue/locks/*.lock  │
│                      │       └──────────────────────────────┘
└──────────────────────┘                       │
            │                                  │ atomic claim
            │ proposes via Sobol/Ax            ▼
            │ stages runfiles per seq          ┌──────────────────────┐
            │ enqueues tasks                   │ N × resident workers │
            ▼                                  │ (long-walltime, 1GPU │
       prepare_candidate                       │  each)               │
       submit_candidate                        └──────────────────────┘
                                                          │
                                                          ▼
                                                   rg_data.txt
                                                   loss.csv
```

### Atomic claims

```python
try:
    os.mkdir(lock_path)
except FileExistsError:
    continue
```

That single syscall is the synchronization primitive. POSIX guarantees
the create-or-fail semantics on a single filesystem, so we get
correctness with zero extra dependencies (no Redis, no ZooKeeper, no
Slurm dependencies). Tasks are JSON files; locks are empty
directories; recovery after a crash is "re-flag stale files".

## Walltime-aware self-requeue

Each worker stops accepting new tasks inside the last
`stop_launch_within_hours` of its allocation, submits a replacement
worker via `record_worker_job`, and exits cleanly so its GPU returns to
the pool with no missed task. Set `requeue_on_stop: true` to enable.

## Startup watchdog

A worker that does not produce its first log line inside
`startup_watchdog_minutes` is treated as stuck (e.g. GPU init failure
or NCCL hang). The controller flags the task PENDING again and the
worker exits so Slurm can re-allocate it. The relevant filenames are
configurable via `resident_worker.startup_watchdog_files`.

## Stale-worker reaper

The controller periodically checks every RUNNING task whose heartbeat
lapsed; if its worker job is no longer in Slurm (`squeue -j <id>`
returns nothing), the task is failed. The BO records this as a failure
in the Gaussian-process noise model rather than blocking on it
forever.

## Streaming vs batch submission

* `bo.submit_mode: streaming` — the moment any one sequence in a
  candidate's batch finishes, the controller proposes a replacement
  candidate. Posterior updates happen *between* the in-flight
  candidates, so you never idle.
* `bo.submit_mode: batch` — the controller waits for the *whole*
  running set to clear before proposing the next batch. Use this when
  you specifically want a synchronous batch-BO step (e.g. you want all
  candidates in a batch to use the same posterior).

The default is `streaming`; switch only if you need synchrony.
