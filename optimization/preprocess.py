"""Sequence-table loading and AA-balanced anchor selection.

The user supplies a sequence table (Excel or CSV) with at least the
following columns:

  * ``label`` — unique sequence id used everywhere downstream.
  * ``Rg``    — reference (e.g. experimental) radius of gyration in nm.
  * ``seq``   — the sequence string. Used to compute composition for
                AA-balanced anchor selection.

Optionally, ``rg_targets`` (a one-Rg-per-line text file) can override the
``Rg`` column at runtime. This is useful when the same sequence file is
shared across campaigns with different reference targets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd


def parse_rg_targets(path: str | Path) -> list[float]:
    values = []
    for line in Path(path).read_text().splitlines():
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        if nums:
            # Convention: file holds Rg in Angstroms, convert to nm.
            values.append(float(nums[-1]) / 10.0)
    return values


def load_sequence_table(config: dict[str, Any]) -> pd.DataFrame:
    """Load the user-supplied sequence table; attach reference Rg + length."""
    seq_path = Path(config["paths"]["sequence_table"])
    if seq_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(seq_path)
    else:
        df = pd.read_csv(seq_path)

    required = {"label", "seq"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Sequence table missing required columns: {sorted(missing)}")

    df = df.copy()
    df["label"] = df["label"].astype(str)
    df["seq"] = df["seq"].astype(str)
    df["length"] = df["seq"].str.len()

    rg_targets_path = config["paths"].get("rg_targets")
    if rg_targets_path:
        rg_values = parse_rg_targets(rg_targets_path)
        if len(df) != len(rg_values):
            raise ValueError(
                f"Rg target count mismatch: {len(df)} sequences vs {len(rg_values)} Rg values"
            )
        df["exp_rg_nm"] = rg_values
    elif "Rg" in df.columns:
        df["exp_rg_nm"] = df["Rg"].astype(float)
    else:
        raise KeyError(
            "Sequence table needs an 'Rg' column or paths.rg_targets must be set"
        )
    return df


def composition(seq: str, vocab: list[str]) -> np.ndarray:
    """Per-bead frequencies (sum to 1) for one sequence."""
    s = seq.upper()
    if not s:
        return np.zeros(len(vocab), dtype=float)
    counts = np.array([s.count(v[0] if len(v) > 1 else v) for v in vocab], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def aa_balanced_anchor(
    table: pd.DataFrame,
    vocab: list[str],
    k: int,
    seed: int = 0,
) -> list[str]:
    """Pick *k* sequences whose pooled composition most closely matches the
    full pool's composition.

    Uses a simple greedy approach: start with the median-Rg sequence, then
    iteratively add the sequence that minimises the L1 distance between the
    selected pool's mean composition and the full pool's mean composition.
    Good enough for a public-facing example; for production the original
    implementation also enforces Rg coverage constraints which we omit here.
    """
    if k <= 0 or len(table) == 0:
        return []
    if k >= len(table):
        return table["label"].astype(str).tolist()

    rng = np.random.default_rng(seed)
    full_comp = np.mean(
        np.stack([composition(seq, vocab) for seq in table["seq"].astype(str)]),
        axis=0,
    )

    # Seed the anchor with the median-Rg sequence.
    sorted_table = table.sort_values("exp_rg_nm").reset_index(drop=True)
    anchor_idx = [int(len(sorted_table) // 2)]

    def selected_mean(indices: list[int]) -> np.ndarray:
        rows = sorted_table.iloc[indices]
        return np.mean(
            np.stack([composition(seq, vocab) for seq in rows["seq"].astype(str)]),
            axis=0,
        )

    for _ in range(k - 1):
        best_idx, best_score = None, float("inf")
        for cand in range(len(sorted_table)):
            if cand in anchor_idx:
                continue
            trial_mean = selected_mean(anchor_idx + [cand])
            score = float(np.abs(trial_mean - full_comp).sum())
            # Tie-break with a small random nudge so different seeds give
            # different but equally-good anchors.
            score += 1e-9 * float(rng.random())
            if score < best_score:
                best_score = score
                best_idx = cand
        if best_idx is None:
            break
        anchor_idx.append(int(best_idx))
    chosen = sorted_table.iloc[sorted(anchor_idx)]
    return chosen["label"].astype(str).tolist()


def build_fixed10_table(config: dict[str, Any], out_path: str | Path | None = None) -> pd.DataFrame:
    """Generate the AA-balanced fixed10 anchor and persist it as a CSV."""
    table = load_sequence_table(config)
    vocab = list(config["amino_acids"])
    k = int(config["objective"]["fixed_per_candidate"])
    seed = int(config["objective"].get("random_seed", 0))
    labels = aa_balanced_anchor(table, vocab, k, seed=seed)
    out = table[table["label"].isin(labels)][["label", "exp_rg_nm", "length"]].copy()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the AA-balanced fixed10 anchor from a sequence table."
    )
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from optimization.common import load_config

    config = load_config(args.config)
    out = args.out or config["paths"].get("fixed10_table") or "fixed10.csv"
    df = build_fixed10_table(config, out)
    print(f"[preprocess] wrote {out} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
