from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from optimization.common import campaign_dir, ensure_dir, load_config


def load_sequence_table(config: dict) -> pd.DataFrame:
    """Load the user-provided sequence/reference table.

    Expected columns: ``label`` (sequence id used everywhere downstream),
    ``Rg`` (reference radius of gyration in nm), ``seq`` (the sequence
    string, retained for record-keeping only).
    """
    path = Path(config["paths"]["sequence_table"])
    df = pd.read_excel(path)
    required = {"label", "Rg", "seq"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Sequence table is missing columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(str)
    return df


def discover_rg_summaries(config: dict) -> pd.DataFrame:
    """Aggregate warm-start ``rg_summary.csv`` files into one frame.

    The glob pattern is configurable so different upstream sweep layouts can
    be plugged in. Each summary file is expected to contain ``label``,
    ``sim_rg_nm``, ``exp_rg_nm``, ``abs_error_nm`` columns. A ``warm_label``
    is attached so observations from different sweeps remain distinguishable.
    """
    paths_cfg = config["paths"]
    data_root = Path(paths_cfg["data_root"])
    glob_pattern = str(paths_cfg.get("warm_start_glob", "*/rg_summary.csv"))
    frames = []
    for path in sorted(data_root.glob(glob_pattern)):
        df = pd.read_csv(path)
        df["warm_label"] = path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No warm-start Rg summaries under {data_root} matching {glob_pattern}"
        )
    return pd.concat(frames, ignore_index=True)


def build_initial_data(config: dict, rg_df: pd.DataFrame) -> pd.DataFrame:
    """Build a warm-start observation table grouped by upstream sweep."""
    selected = set(config["selected_sequences"])
    subset = rg_df[rg_df["label"].isin(selected)].copy()
    grouped = []
    for warm_label, group in subset.groupby("warm_label"):
        if set(group["label"]) != selected:
            continue
        errors = group["sim_rg_nm"].to_numpy() - group["exp_rg_nm"].to_numpy()
        grouped.append(
            {
                "trial_key": warm_label,
                "source": "warm_start",
                "loss": float((errors**2).mean()),
                "mae_nm": float(abs(errors).mean()),
                "rmse_nm": float(((errors**2).mean()) ** 0.5),
                "n_sequences": int(len(group)),
                "noise_sd": float(config["objective"].get("warm_start_noise", 0.25)),
            }
        )
    return pd.DataFrame(grouped).sort_values("loss").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate warm-start results into BO-ready data.")
    parser.add_argument("--config", default="config.example.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(campaign_dir(config) / "preprocessed")
    rg_df = discover_rg_summaries(config)
    initial_df = build_initial_data(config, rg_df)

    rg_df.to_csv(out_dir / "all_rg_summary.csv", index=False)
    initial_df.to_csv(out_dir / "initial_data.csv", index=False)

    print(f"[preprocess] warm start -> {out_dir / 'initial_data.csv'}")
    print(initial_df.to_string(index=False))


if __name__ == "__main__":
    main()
