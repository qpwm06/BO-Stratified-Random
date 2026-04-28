# Stratified-random subsampling

## The problem

Each Bayesian-optimization candidate is a point in a *D*-dimensional
parameter space. Evaluating that candidate means running *N* expensive
simulations — one per sequence in the reference pool — and then
aggregating their per-sequence errors into a single scalar loss. With
*N* in the low hundreds and each simulation taking tens of minutes on
a GPU, a full BO campaign would burn through compute long before the
posterior converges.

So we do not evaluate the full pool. We evaluate a **stratified-random
subsample** instead.

## The idea

Per candidate we run *K* + *K* simulations: *K* "fixed" sequences shared
across the whole campaign, and *K* "random" sequences re-drawn for every
candidate. The proxy loss is

```
loss(candidate) = w_fixed · MAE_fixed(candidate) + w_random · MAE_random(candidate)
```

with `w_fixed + w_random = 1`. In the bundled config we use `K = 10`
and equal weights, so each candidate runs **20** simulations instead of
*N* — a ~6.85× reduction at *N* = 137.

### Why it is unbiased

* **Fixed anchor.** The *K* fixed sequences are picked once at the start
  of the campaign so their pooled amino-acid composition matches the
  pool's composition (`optimization/preprocess.py::aa_balanced_anchor`).
  They contribute the *same* sequences to every candidate, which gives
  the BO posterior a low-variance reference signal.
* **Stratified random tail.** The *K* random sequences are drawn
  per-candidate from the remaining *N − K* sequences. The sampler in
  `optimization/sampling.py::sample_random_labels` partitions the
  remaining pool into 10 quantile bins by reference R$_g$, then forces
  each candidate's random subset to touch as many bins as possible while
  also avoiding collisions inside a batch.

Because the random tail rotates and the anchor is fixed, the expectation
of the proxy MAE over many candidates equals the MAE that would have
been measured on the full pool. The variance is bounded by the bin
sizes, which is why we stratify rather than sample uniformly.

### Validation

We re-ran the top-10 BO candidates on the full *N*-sequence pool. The
proxy loss tracks the full-pool MAE almost perfectly — see
[results_release/top5_vs_full.png](../results_release/top5_vs_full.png)
and [docs/RESULTS.md](RESULTS.md) for the numbers. Splitting Sobol from
BoTorch shows the same pattern: the candidate the proxy thinks is best
also wins on the full pool.

## Knobs

The behaviour above is controlled by `objective` in
`config.example.yaml`:

```yaml
objective:
  fixed_per_candidate:  10   # K
  random_per_candidate: 10   # K
  fixed_weight:        0.5   # w_fixed
  random_weight:       0.5   # w_random
  huber_delta:        0.25   # 0 disables; soft-caps per-sequence error
  burnin_ns:           5.0   # discarded from each Rg trace
  random_seed:    20260425   # rotates the per-batch random subsets
```

Set `huber_delta > 0` if a few sequences in your pool are pathological
outliers; the soft cap stops them from dominating the proxy loss.

## Practical notes

* If you change the sequence pool, regenerate the fixed10 anchor before
  resuming a campaign:
  `python -m optimization.preprocess --config <your-config>`.
* The stratified scheme assumes the per-sequence error is
  approximately stationary in the parameter space the BO walks through.
  If your simulator is bimodal (e.g. some parameter region kills a
  subset of sequences), the random tail can mask the regression. Inspect
  `fixed_mae_nm` and `random_mae_nm` separately in `observations.csv`
  if you suspect this.
* The fixed anchor is picked greedily on amino-acid composition; we did
  *not* enforce R$_g$ coverage. For most pools this is fine. If your
  pool has a heavy R$_g$ skew you may want to extend
  `preprocess.aa_balanced_anchor` with an Rg-coverage tie-break.
