"""Weighted fixed + random loss for the stratified-random BO.

The candidate's per-sequence Rg results split into two groups:

* ``fixed_mae`` — mean absolute error on the K anchored sequences (low
  variance across candidates because the anchor is shared).
* ``random_mae`` — mean absolute error on the K stratified-random
  sequences (high variance across candidates, but unbiased over many).

The combined loss is a convex combination, with weights set in the config
(``objective.fixed_weight`` and ``objective.random_weight``; both default
to 0.5). A Huber soft cap can optionally be applied per-sequence to reduce
the influence of outliers — see ``objective.huber_delta``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def huber(error: float, delta: float) -> float:
    """Standard Huber loss; quadratic near 0, linear far away."""
    error = abs(float(error))
    if delta <= 0:
        return error * error
    if error <= delta:
        return 0.5 * error * error
    return delta * (error - 0.5 * delta)


def attach_reference(config: dict, rg_df: pd.DataFrame, sequence_table: pd.DataFrame) -> pd.DataFrame:
    """Join simulated Rg per sequence with the reference Rg from the table."""
    table = sequence_table[["label", "exp_rg_nm"]].copy()
    merged = rg_df.merge(table, on="label", how="left")
    if merged["exp_rg_nm"].isna().any():
        missing = merged.loc[merged["exp_rg_nm"].isna(), "label"].tolist()
        raise KeyError(f"Missing reference Rg for labels: {missing}")
    return merged


def compute_loss(
    config: dict[str, Any],
    rg_df: pd.DataFrame,
    sequence_table: pd.DataFrame,
    fixed_labels: list[str],
) -> tuple[float, dict[str, float], pd.DataFrame]:
    """Stratified loss = ``fixed_weight * fixed_MAE + random_weight * random_MAE``.

    Returns the scalar loss, a dict of per-component metrics, and the
    enriched per-sequence detail frame.
    """
    objective = config["objective"]
    merged = attach_reference(config, rg_df, sequence_table)
    merged["abs_error_nm"] = (merged["sim_rg_nm"] - merged["exp_rg_nm"]).abs()

    delta = float(objective.get("huber_delta", 0.0))
    if delta > 0:
        merged["loss_per_seq"] = merged["abs_error_nm"].apply(lambda e: huber(e, delta))
    else:
        merged["loss_per_seq"] = merged["abs_error_nm"]

    fixed_set = set(fixed_labels)
    fixed_part = merged[merged["label"].isin(fixed_set)]
    random_part = merged[~merged["label"].isin(fixed_set)]

    fixed_mae = float(fixed_part["loss_per_seq"].mean()) if not fixed_part.empty else float("nan")
    random_mae = float(random_part["loss_per_seq"].mean()) if not random_part.empty else float("nan")

    fw = float(objective.get("fixed_weight", 0.5))
    rw = float(objective.get("random_weight", 0.5))
    if np.isnan(fixed_mae):
        loss = random_mae
    elif np.isnan(random_mae):
        loss = fixed_mae
    else:
        loss = fw * fixed_mae + rw * random_mae

    metrics = {
        "loss": loss,
        "fixed_mae_nm": fixed_mae,
        "random_mae_nm": random_mae,
        "n_fixed": int(len(fixed_part)),
        "n_random": int(len(random_part)),
    }
    return loss, metrics, merged


def write_loss_outputs(
    config: dict,
    candidate_dir: str | Path,
    rg_df: pd.DataFrame,
    sequence_table: pd.DataFrame,
    fixed_labels: list[str],
) -> dict[str, float]:
    candidate_path = Path(candidate_dir)
    loss, metrics, detail = compute_loss(config, rg_df, sequence_table, fixed_labels)
    detail.to_csv(candidate_path / "sequence_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(candidate_path / "loss.csv", index=False)
    return metrics
