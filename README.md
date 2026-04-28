# BO-Stratified-Random

Asynchronous Bayesian optimization for high-dimensional parameter
sweeps. Instead of running every sequence in the reference pool for
every BO candidate, this controller runs a **stratified-random
subsample** (10 fixed AA-balanced anchor + 10 Rg-stratified random
sequences) per candidate. Validation against the full pool confirms the
proxy is unbiased; the compute saving is roughly **6.85×** at *N* = 137
sequences.

A persistent Slurm worker pool consumes a local on-disk task queue, so
the only Slurm allocations the cluster scheduler ever sees are *N*
long-walltime workers — no per-simulation `sbatch` overhead, no queue
tax between batches.

## At a glance

* **Stratified-random subsampling** — `K` fixed (anchored) + `K` random
  (rotating, Rg-stratified) per candidate. Unbiased over many
  candidates, six times cheaper per candidate than the full sweep.
  Defaults: `K = 10`, equal weights, optional Huber soft-cap.
* **Sobol → BoTorch handoff** — first `n_initial_sobol` (default 60)
  candidates explore via Sobol. After that, Ax/BoTorch (qLogNEI) takes
  over.
* **Resident Slurm worker pool** — atomic `mkdir` lock claims, walltime
  self-requeue, startup watchdog, stale-worker reaper.
* **Three launchers** — `direct` (one `sbatch` per simulation),
  `resident` (queue + worker pool, the production setting), and
  `local` (synchronous in-process, for the bundled fake runner).
* **Drop-in fake simulator** — `optimization/fake_run.py` lets you
  exercise the BO loop end-to-end without GPUs or MD. Useful for
  smoke-testing the orchestration before wiring up your engine.

## Sample results

Best-so-far loss across one real campaign, with the Sobol exploration
band (gray) and the BoTorch optimization band (blue) highlighted:

![BO loss curve](results_release/observations_curve.png)

The top-5 BO candidates were re-evaluated on the *full* sequence pool
to confirm the proxy is unbiased:

![Top 5 vs full](results_release/top5_vs_full.png)

The "before" we care about is two parameter tables derived from two
**different physical priors** — the natural baselines you would use
without optimisation, not an early Sobol candidate. All three panels
below are evaluated on the full sequence pool:

![Before vs after](results_release/before_after.png)

| panel               | full-pool MAE | Pearson r |
|---------------------|---------------|-----------|
| Physical baseline A | 1.061 nm      | 0.51      |
| Physical baseline B | 1.864 nm      | 0.22      |
| Best BO             | 0.623 nm      | 0.74      |

See [docs/RESULTS.md](docs/RESULTS.md) for the per-panel interpretation
and a writeup of the CVaR-80 + warm-start follow-up campaign.

## Quickstart with the fake runner

No GPU, no Slurm, no MD engine required.

```bash
# 1. Install dependencies.
python -m pip install -r requirements.txt

# 2. Make a sequence table at ./templates/sequences.xlsx with columns:
#       label, seq, Rg
#    Drop one runfile/<label>/ template per sequence under ./templates/.

# 3. Edit config.example.yaml:
#       slurm.launcher:        local
#       paths.simulation_python: ./scripts/fake_python.sh
#    where fake_python.sh is one line:
#       #!/usr/bin/env bash
#       exec python -m optimization.fake_run

# 4. Generate the AA-balanced fixed10 anchor.
python -m optimization.preprocess --config config.example.yaml --out fixed10.csv

# 5. Run the controller.
python -m optimization.bo_loop --config config.example.yaml --once   # one pass
python -m optimization.bo_loop --config config.example.yaml --once   # second pass collects loss
```

After the second `--once` pass you should see a populated `runs/<campaign>/observations.csv`,
a `loss.csv` per candidate, and an updated `performance.png` with the
Sobol/BO bands.

## Wiring up your own simulator

There is exactly one project-specific function you need to fill in:

```python
# optimization/patch_sim.py
def apply_user_patch_hook(run_dir, config, parameters):
    # rewrite run_dir/production.py so it uses ``parameters`` instead of
    # the template's defaults, then return.
    ...
```

The runfile copy, SBATCH script generation, queue mechanics, status
polling, loss computation, and BO update are all unchanged. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data flow.

## Documentation

* **[docs/STRATIFIED_SAMPLING.md](docs/STRATIFIED_SAMPLING.md)** —
  the math behind the proxy loss and the validation results.
* **[docs/SLURM_ORCHESTRATION.md](docs/SLURM_ORCHESTRATION.md)** —
  resident workers, atomic locks, walltime requeue, watchdogs.
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — diagram and
  module map.
* **[docs/RESULTS.md](docs/RESULTS.md)** — annotated plots + headline
  numbers from the bundled sample campaign.
* **[baseline/README.md](baseline/README.md)** — the earlier vanilla BO
  controller, kept under `baseline/` so you can side-by-side benchmark
  the stratified-random version against the run-them-all approach.

## Repository layout

```
optimization/             # stratified-random BO (the primary controller)
baseline/                 # earlier vanilla BO (kept for reference)
config.example.yaml       # sanitized config; copy and edit per campaign
scripts/run_bo_controller.sh
slurm/job.template.slurm
docs/                     # the four docs above
results_release/          # anonymised sample results
tools/anonymize_results.py
```

## Acknowledgements

This project is a collaboration between the author and two AI coding
assistants, with clearly separated roles:

* **Author** — designed the overall framework (the stratified-random
  proxy, the Sobol → BoTorch handoff, the resident-worker /
  task-queue architecture, the validation-campaign protocol), ran all
  the molecular-dynamics simulations that produced the sample results
  in `results_release/`, and directed the AI tools throughout.
* **[Codex](https://platform.openai.com/docs/codex)** (OpenAI) —
  implemented the BO controller (`optimization/bo_loop.py`,
  `loss.py`, `preprocess.py`, `monitor.py`, `stratified_validation.py`)
  and the resident-worker / Slurm-orchestration layer
  (`resident_worker.py`, `slurm_driver.py`) under the author's
  guidance.
* **[Claude Code](https://www.anthropic.com/claude-code)** (Anthropic)
  — produced this public release: split the long internal README into
  topic-specific `docs/` pages, wrote the anonymization tooling
  (`tools/anonymize_results.py`), regenerated the figures with the
  Sobol/BO bands and the physical baselines, and scrubbed the codebase
  of identity-leaking paths and identifiers.

## License

Released under the [MIT License](LICENSE).
