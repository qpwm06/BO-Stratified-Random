# Vanilla BO baseline

This directory holds the **earlier, vanilla Bayesian-optimization** controller
that the stratified-random version under `../optimization/` replaces.

The vanilla baseline:

* Proposes candidates via Ax (BoTorch qLogNEI) with a Sobol fallback.
* Evaluates **every selected sequence** for every BO candidate.
* Computes a single MSE loss per candidate over those sequences.

It is included for reference and reproducibility — to make it possible to
side-by-side benchmark the stratified-random strategy against the
"run-them-all" approach. None of the resident-worker / Slurm-orchestration
glue code is duplicated here; that lives in `../optimization/` and is shared
by both controllers.

## Contents

* `bo_loop.py` — the vanilla async BO controller.
* `loss.py` — single-MSE-per-candidate loss.
* `preprocess.py` — warm-start aggregation from upstream sweeps.

## Running it

The baseline controller imports from the same `../optimization` package
(monitor, resident_worker, slurm_driver, patch_sim, etc.). To run it after
you have cloned the repository, point your `PYTHONPATH` at the repo root
and invoke:

```bash
python -m baseline.bo_loop --config config.example.yaml
```

(You may need to tweak `paths.simulation_python` etc. in the config the
same way you would for the stratified controller.)
