from __future__ import annotations

from pathlib import Path

import pandas as pd

from baseline.preprocess import load_sequence_table


def attach_reference(config: dict, rg_df: pd.DataFrame) -> pd.DataFrame:
    """Join simulated Rg per sequence with the reference (experimental) Rg."""
    table = load_sequence_table(config)[["label", "Rg"]].rename(columns={"Rg": "exp_rg_nm"})
    merged = rg_df.merge(table, on="label", how="left")
    if merged["exp_rg_nm"].isna().any():
        missing = merged.loc[merged["exp_rg_nm"].isna(), "label"].tolist()
        raise KeyError(f"Missing reference Rg for: {missing}")
    return merged


def compute_loss(config: dict, rg_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    merged = attach_reference(config, rg_df)
    merged["error_nm"] = merged["sim_rg_nm"] - merged["exp_rg_nm"]
    merged["abs_error_nm"] = merged["error_nm"].abs()
    merged["sq_error_nm2"] = merged["error_nm"] ** 2
    metric = config["objective"].get("metric", "mse")
    if metric != "mse":
        raise ValueError(f"Only MSE is implemented in the public release: {metric}")
    return float(merged["sq_error_nm2"].mean()), merged


def write_loss_outputs(config: dict, candidate_dir: str | Path, rg_df: pd.DataFrame) -> float:
    candidate_path = Path(candidate_dir)
    loss, detail = compute_loss(config, rg_df)
    detail.to_csv(candidate_path / "sequence_results.csv", index=False)
    pd.DataFrame([{"loss": loss, "mae_nm": detail["abs_error_nm"].mean(), "rmse_nm": loss**0.5}]).to_csv(
        candidate_path / "loss.csv", index=False
    )
    return loss
