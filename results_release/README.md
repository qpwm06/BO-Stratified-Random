# Sample BO results

Representative slice of a real campaign, anonymised for public release.
Sequence labels have been replaced with ``seqXXX`` indices and parameter
values are not exposed.

## Files

* `observations_curve.png` — proxy loss vs. candidate index for the live
  stratified-random BO. Background bands mark the Sobol exploration
  region (``[0, 60)``) and the BoTorch BO region
  (``[60, 100)``). The legend is mounted
  above the plot (``ncol=3``) so it never occludes the data.

* `before_after.png` — three-panel comparison, all evaluated on the
  *full* sequence pool: physical baseline A (left), physical baseline B
  (middle), best BO candidate (right). The two baselines correspond to
  parameter sets derived from two different physical priors.

* `top5_vs_full.png` — top-5 BO candidates from the stratified campaign,
  re-evaluated on the full sequence pool. Confirms that the
  fixed10 + random10 proxy is an unbiased estimator of the full-pool
  MAE.

* `observations_anon.csv` — proxy-loss history with parameter columns
  stripped. Columns: ``candidate_index``, ``status``, ``loss``,
  ``fixed_mae_nm``, ``random_mae_nm``, ``phase``.

* `candidates_examples/` — 3 candidate folders.
  Each holds:
    * ``params.json`` — only the parameter-vector dimensionality
      (numeric values are intentionally omitted).
    * ``loss.csv`` — full-pool loss summary for the candidate.

## Headline numbers

* Trials: **100** stratified candidates, **60**
  in the Sobol exploration phase and **40**
  in the BoTorch optimization phase.
* Physical baseline A full-pool MAE: **1.061 nm**.
* Physical baseline B full-pool MAE: **1.864 nm**.
* Best BO            full-pool MAE: **0.623 nm**.
