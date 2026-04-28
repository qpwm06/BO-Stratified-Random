# Architecture

```
                 ┌──────────────────────────────────────┐
                 │      bo_loop.py (controller)         │
                 │                                      │
                 │   1. propose candidate               │
                 │      • Sobol while                   │
                 │        len(obs) < n_initial_sobol    │
                 │      • Ax/BoTorch (qLogNEI) after    │
                 │   2. choose sequences (sampling.py)  │
                 │      • K fixed AA-balanced anchor    │
                 │      • K stratified-random tail      │
                 │   3. stage runfile per sequence      │
                 │   4. enqueue tasks                   │
                 └──────────────────────┬───────────────┘
                                        │
                  ┌─────────────────────┼─────────────────────┐
                  │                     │                     │
                  ▼                     ▼                     ▼
         ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
         │ direct launcher   │   │ resident launcher │   │  local launcher   │
         │ (sbatch per task) │   │ (queue + workers) │   │ (run synchronously)│
         └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                   │                       │                       │
                   ▼                       ▼                       ▼
                                   ┌──────────────────────────────────┐
                                   │  rg_data.txt for each sequence   │
                                   └──────────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────────┐
                                   │  loss.compute_loss               │
                                   │  fixed-anchor MAE + random MAE   │
                                   │   → loss + per-sequence detail   │
                                   └──────────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────────┐
                                   │  monitor.refresh_monitor_outputs │
                                   │   • status.csv                   │
                                   │   • performance.png              │
                                   │     (Sobol + BO regions)         │
                                   │  state.json + observations.csv   │
                                   └──────────────────────────────────┘
                                                  │
                                                  ▼
                                          back to step 1
```

## Module map

```
optimization/
  bo_loop.py            # async controller (this file orchestrates everything)
  sampling.py           # stratified-random subsampling (10 fixed + 10 random)
  preprocess.py         # sequence-table loader + AA-balanced anchor selection
  loss.py               # weighted MAE on fixed + random subsets, optional Huber
  monitor.py            # status.csv + performance.png with Sobol/BO bands
  patch_sim.py          # apply_user_patch_hook stub (engine-specific patcher)
  fake_run.py           # synthetic simulator for the smoke test
  rg_compute.py         # placeholder for trajectory loaders
  rg_parser.py          # rg_data.txt -> mean / SE / span
  slurm_driver.py       # direct / resident / local launchers
  resident_worker.py    # persistent worker pool + atomic task queue
  common.py             # config, paths, atomic JSON writes

baseline/               # earlier vanilla BO (no stratified subsampling)
  bo_loop.py
  loss.py
  preprocess.py

config.example.yaml     # sanitized config (placeholders for cluster paths)
docs/                   # the four docs
results_release/        # anonymised sample results
tools/anonymize_results.py
```

## State machine

The controller persists **everything** to `runs/<campaign_name>/`:

* `state.json`        — next candidate index, batch index, fixed10 labels,
                         per-candidate metadata (`status`, `loss`,
                         `fixed_mae_nm`, `random_mae_nm`, `labels`,
                         `lambda_pw`, `source`)
* `observations.csv`  — flat per-candidate history (this is what the
                         results plots consume)
* `status.csv`        — current snapshot of running candidates and the
                         resident worker pool
* `performance.png`   — the Sobol/BO loss curve described in
                         [docs/RESULTS.md](RESULTS.md)
* `candidates/<cand>/` — per-candidate `params.json`, `loss.csv`,
                         `sequence_results.csv`, runfile tree
* `resident_queue/`   — `tasks/*.json`, `locks/*.lock`, `worker_pool.json`

If the controller crashes mid-campaign, restarting it picks up the
existing state and continues. The same `state.json` is what the
anonymizer at `tools/anonymize_results.py` reads to produce the
release plots.
