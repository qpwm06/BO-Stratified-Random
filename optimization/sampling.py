"""Stratified-random subsampling of the sequence pool.

The full evaluation cost of one BO candidate is *N* MD simulations, where *N*
is the size of the sequence pool. When *N* is large (e.g. 137), evaluating
every candidate on the full pool is wasteful — most of the variance in the
posterior comes from a handful of high-leverage sequences.

This module implements the **fixed-K + random-K** trick:

* ``fixed_per_candidate`` (default 10) sequences are picked once for the whole
  campaign, balanced so their amino-acid composition reproduces the full
  pool's composition (see ``preprocess.aa_balanced_anchor``). They're shared
  across candidates and provide a low-variance "anchor" comparison.
* ``random_per_candidate`` (default 10) additional sequences are drawn fresh
  for every candidate, **stratified across Rg quantile bins** so the random
  subset always covers the full range of target sizes. They give the proxy
  loss its Monte-Carlo unbiasedness over many candidates.

The proxy loss is then a weighted average of the MAE on the fixed and random
subsets (see ``loss.compute_loss``). Over many candidates, ``E[L_proxy]``
converges to the true full-pool loss; the ratio ``random_per_candidate / N``
controls the variance, the ratio ``fixed_per_candidate / N`` shifts the bias.

The cost saving is dramatic: with ``fixed=10, random=10`` and ``N=137``, each
candidate runs only 20 simulations instead of 137 — about 6.85× cheaper —
and our validation campaign confirms there is no detectable bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rg_quantile_bins(table: pd.DataFrame, n_bins: int = 10) -> pd.Series:
    """Return a per-row Rg-quantile bin id (0 .. n_bins-1)."""
    return pd.qcut(table["exp_rg_nm"], n_bins, labels=False, duplicates="drop")


def sample_random_labels(
    config: dict[str, Any],
    table: pd.DataFrame,
    fixed: set[str],
    batch_index: int,
    n_candidates: int,
) -> dict[int, list[str]]:
    """Stratified-random label sampler.

    For each candidate in a batch of ``n_candidates``, pick
    ``random_per_candidate`` sequences from the pool *excluding* the fixed
    anchor. The selection is stratified across Rg-quantile bins so the
    random subset always covers the full range of target sizes; ties are
    broken so labels are unique within the batch (no collisions across the
    candidates of the same batch).

    Returns a mapping ``{candidate_offset: [label, ...]}``.
    """
    rng = np.random.default_rng(int(config["objective"]["random_seed"]) + batch_index)
    random_per_candidate = int(config["objective"]["random_per_candidate"])
    available = table[~table["label"].isin(fixed)].copy()
    available["rg_bin"] = rg_quantile_bins(available, 10)
    bins = [int(b) for b in sorted(available["rg_bin"].dropna().unique())]
    if not bins:
        raise RuntimeError("No Rg bins available for random sampling")

    # Track labels already used in this batch so different candidates get
    # disjoint random subsets (cuts batch-level variance further).
    selected: set[str] = set()
    label_to_bin = dict(zip(available["label"].astype(str), available["rg_bin"].astype(int)))
    out: dict[int, list[str]] = {}

    for offset in range(n_candidates):
        candidate_labels: list[str] = []
        shuffled = bins.copy()
        rng.shuffle(shuffled)
        for bin_id in shuffled:
            if len(candidate_labels) >= random_per_candidate:
                break
            group = available[
                (available["rg_bin"] == bin_id) & (~available["label"].isin(selected))
            ]
            if group.empty:
                continue
            label = str(group.iloc[int(rng.integers(0, len(group)))]["label"])
            candidate_labels.append(label)
            selected.add(label)
        # If we ran out of unique-per-bin slots (e.g. random_per_candidate >
        # n_bins or batch already consumed a bin), refill from whatever's
        # left, prioritising bins that still have many remaining options.
        while len(candidate_labels) < random_per_candidate:
            remaining = available[~available["label"].isin(selected)]
            if remaining.empty:
                raise RuntimeError(
                    "Not enough unique labels for batch random sampling; "
                    "reduce random_per_candidate or batch_size."
                )
            bin_counts = remaining["rg_bin"].value_counts()
            refill_bins = sorted(bin_counts.index, key=lambda b: (-int(bin_counts.loc[b]), int(b)))
            for bin_id in refill_bins:
                group = remaining[remaining["rg_bin"] == bin_id]
                if group.empty:
                    continue
                label = str(group.iloc[int(rng.integers(0, len(group)))]["label"])
                candidate_labels.append(label)
                selected.add(label)
                break
            else:
                raise RuntimeError("Could not refill random labels from remaining bins")
        rng.shuffle(candidate_labels)
        out[offset] = candidate_labels
        # Sanity: when we have at least as many bins as random_per_candidate,
        # every candidate's random subset must touch as many distinct bins.
        covered = {int(label_to_bin[label]) for label in candidate_labels}
        if len(bins) >= random_per_candidate and len(covered) < random_per_candidate:
            raise RuntimeError(
                f"Candidate {offset} random labels do not cover {random_per_candidate} Rg bins"
            )
    return out


def labels_for_candidate(
    config: dict[str, Any],
    table: pd.DataFrame,
    fixed: list[str],
    batch_index: int,
    candidate_offset: int,
) -> list[str]:
    """Convenience wrapper: fixed anchor + this candidate's stratified-random subset."""
    n_candidates = int(config["bo"].get("batch_size", 1))
    random_map = sample_random_labels(config, table, set(fixed), batch_index, n_candidates)
    return list(fixed) + random_map[candidate_offset]
