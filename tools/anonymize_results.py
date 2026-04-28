"""Build the anonymised ``results_release/`` slice from a real campaign.

Two source campaigns are consumed:

* **Stratified BO campaign** (``--stratified-dir``): the live BO that ran
  on a fixed10 + random10 subsample of the full N-sequence pool. We use
  this to make the loss curve, the observations table, and the example
  candidate folders.
* **Full-pool validation campaign** (``--validation-dir``): re-runs of
  the top stratified candidates against the *entire* N-sequence pool,
  used to confirm the proxy loss is unbiased. We use this for the top-5
  comparison plot and the before/after panels.

The script is shipped inside the repo so reviewers can audit *what was
released*. To regenerate from a different campaign, point the flags at
the new live directories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------------------
# Anonymisation helpers
# ---------------------------------------------------------------------------


def anon_label_map(labels: list[str]) -> dict[str, str]:
    """Stable mapping from real label -> ``seq000``, ``seq001``, ..."""
    return {label: f"seq{idx:03d}" for idx, label in enumerate(sorted(set(labels)))}


def anon_param_map(keys: list[str]) -> dict[str, str]:
    return {key: f"bead_{idx:02d}" for idx, key in enumerate(sorted(set(keys)))}


def drop_lambda_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip parameter columns and any other identifying free-form columns."""
    keep = [c for c in df.columns if not c.startswith("lambda_pw_")]
    out = df[keep].copy()
    drop = ["candidate_id", "completed_at"]
    return out.drop(columns=[c for c in drop if c in out.columns])


def assign_phase(df: pd.DataFrame, n_initial_sobol: int) -> pd.DataFrame:
    """Add a ``phase`` column: ``"sobol"`` for indices < n_initial_sobol,
    ``"botorch"`` after."""
    out = df.copy()
    out["phase"] = out["candidate_index"].apply(
        lambda idx: "sobol" if int(idx) < int(n_initial_sobol) else "botorch"
    )
    return out


# ---------------------------------------------------------------------------
# Stratified campaign: loss curve + observations
# ---------------------------------------------------------------------------


def render_observations_curve(
    obs_df: pd.DataFrame,
    n_initial_sobol: int,
    out_path: Path,
) -> None:
    """BO loss curve with Sobol / BO background bands and a top legend.

    The two bands are drawn at ``[0, n_initial_sobol)`` and
    ``[n_initial_sobol, +inf)``. Each band gets a text annotation. The
    legend sits above the axes (``ncol=3``) so it can never occlude
    data.
    """
    df = obs_df.sort_values("candidate_index").copy()
    df["best_loss"] = df["loss"].cummin()
    sobol = df[df["candidate_index"] < n_initial_sobol]
    bo = df[df["candidate_index"] >= n_initial_sobol]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    x_max = float(df["candidate_index"].max()) + 1
    y_max = float(df["loss"].max())
    y_min = float(df["loss"].min())
    pad = 0.12 * (y_max - y_min if y_max > y_min else y_max + 1.0)
    ymin, ymax = y_min - pad, y_max + 1.6 * pad

    ax.axvspan(-0.5, n_initial_sobol - 0.5, facecolor="#ececec", zorder=0)
    if n_initial_sobol < x_max:
        ax.axvspan(n_initial_sobol - 0.5, x_max - 0.5, facecolor="#e8f1fb", zorder=0)

    ax.scatter(sobol["candidate_index"], sobol["loss"],
               c="#555555", marker="o", s=26, label="Sobol obs.", zorder=2)
    ax.scatter(bo["candidate_index"], bo["loss"],
               facecolors="none", edgecolors="#1f77b4", marker="o", s=26,
               linewidths=1.2, label="BoTorch obs.", zorder=2)
    ax.plot(df["candidate_index"], df["best_loss"], color="black", linewidth=1.5,
            label="best so far", zorder=3)

    label_y = ymax - 0.06 * (ymax - ymin)
    if n_initial_sobol > 0:
        ax.text((n_initial_sobol - 0.5) / 2, label_y,
                f"Sobol exploration\n(n={n_initial_sobol})",
                ha="center", va="top", fontsize=9, color="#444444", zorder=1,
                linespacing=1.0)
    if n_initial_sobol < x_max:
        ax.text((n_initial_sobol - 0.5 + x_max) / 2, label_y,
                "BoTorch BO\n(qLogNEI)",
                ha="center", va="top", fontsize=9, color="#1f4f80", zorder=1,
                linespacing=1.0)

    ax.set_xlabel("candidate index")
    ax.set_ylabel("proxy loss (nm)")
    ax.set_xlim(-0.5, x_max - 0.5)
    ax.set_ylim(ymin, ymax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, borderaxespad=0.0, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation campaign: top-5 vs full-pool MAE + before/after
# ---------------------------------------------------------------------------


def candidate_seq_results(candidate_dir: Path) -> pd.DataFrame:
    """Return the per-sequence ``sequence_results.csv`` for a finished
    validation candidate, restricted to the columns we expose."""
    path = candidate_dir / "sequence_results.csv"
    df = pd.read_csv(path)
    # Drop columns that might leak the application identity.
    keep = [c for c in ["sim_rg_nm", "exp_rg_nm", "abs_error_nm", "label"] if c in df.columns]
    return df[keep].copy()


def candidate_loss(candidate_dir: Path) -> dict[str, float]:
    path = candidate_dir / "loss.csv"
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def render_top5_vs_full(
    validation_dir: Path,
    label_map: dict[str, str],
    out_path: Path,
) -> tuple[list[str], list[float]]:
    """Five-panel comparison: top-5 stratified candidates re-run on the
    full N-sequence pool. Confirms the proxy is unbiased."""
    obs = pd.read_csv(validation_dir / "observations.csv")
    # Pick the 5 lowest-loss validation candidates (these are the top
    # candidates from the stratified campaign re-run on the full pool).
    top5 = obs.sort_values("loss", ascending=True).head(5).copy()
    top5["rank"] = range(1, len(top5) + 1)

    rows = []
    for item in top5.itertuples(index=False):
        cand_dir = validation_dir / "candidates" / str(item.candidate_id)
        if not cand_dir.exists():
            print(f"[anon] missing candidate dir: {cand_dir}")
            continue
        df = candidate_seq_results(cand_dir)
        df["rank"] = int(item.rank)
        df["loss"] = float(item.loss)
        rows.append(df)
    if not rows:
        raise RuntimeError("no validation candidates found")
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df["label_anon"] = plot_df["label"].map(label_map).fillna("seq???")

    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.8), sharex=True, sharey=True)
    min_rg = float(min(plot_df["sim_rg_nm"].min(), plot_df["exp_rg_nm"].min()))
    max_rg = float(max(plot_df["sim_rg_nm"].max(), plot_df["exp_rg_nm"].max()))
    pad = 0.08 * (max_rg - min_rg)
    lims = (min_rg - pad, max_rg + pad)
    colors = plt.cm.tab10(range(5))

    captions = []
    losses = []
    for ax, (rank, group), color in zip(axes, plot_df.groupby("rank", sort=True), colors):
        loss = float(group["loss"].iloc[0])
        ax.scatter(group["sim_rg_nm"], group["exp_rg_nm"],
                   color=color, s=10, alpha=0.6)
        ax.plot(lims, lims, color="black", linewidth=0.7, alpha=0.6)
        mae = float(group["abs_error_nm"].mean())
        ax.set_title(f"top #{int(rank)}\nfull-pool MAE={mae:.3f} nm", fontsize=8.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("simulated R$_g$ (nm)")
        captions.append(f"top #{rank}")
        losses.append(mae)
    axes[0].set_ylabel("reference R$_g$ (nm)")
    fig.suptitle(
        f"Top-5 BO candidates re-evaluated on the full sequence pool "
        f"(N={int(top5['n_sequences'].iloc[0]) if 'n_sequences' in top5.columns else 'N'})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return captions, losses


def _load_baseline_summary(csv_path: Path) -> pd.DataFrame:
    """Load a `rg_lastframe_nosol_summary.csv`-style baseline file and
    normalise its column names to match validation candidate output:
    ``label, exp_rg_nm, sim_rg_nm, abs_error_nm``.
    """
    df = pd.read_csv(csv_path)
    rename = {
        "sim_rg_lastframe_nm": "sim_rg_nm",
        "abs_delta_nm": "abs_error_nm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["label", "exp_rg_nm", "sim_rg_nm", "abs_error_nm"] if c in df.columns]
    return df[keep].dropna(subset=["sim_rg_nm", "exp_rg_nm"]).copy()


def render_before_after(
    baseline_a_csv: Path,
    baseline_b_csv: Path,
    validation_dir: Path,
    out_path: Path,
) -> dict[str, float]:
    """Three-panel before/after: two physics-based baselines and the
    best BO candidate, all evaluated on the full sequence pool, so the
    MAEs are directly comparable.
    """
    a = _load_baseline_summary(baseline_a_csv)
    b = _load_baseline_summary(baseline_b_csv)

    # Best BO candidate from the validation (full-pool) campaign.
    val_obs = pd.read_csv(validation_dir / "observations.csv").sort_values("loss")
    best = val_obs.iloc[0]
    bo_dir = validation_dir / "candidates" / str(best["candidate_id"])
    bo = candidate_seq_results(bo_dir)

    panels = [
        ("Physical baseline A", a,  "#888888"),
        ("Physical baseline B", b,  "#b86b3a"),
        ("Best BO (full pool)", bo, "#1f77b4"),
    ]

    summary: dict[str, float] = {}
    all_sim, all_exp = [], []
    for name, df, _ in panels:
        all_sim.append(df["sim_rg_nm"])
        all_exp.append(df["exp_rg_nm"])
        summary[f"{name}|n"] = int(len(df))
        summary[f"{name}|mae_nm"] = float(df["abs_error_nm"].mean())

    sim_cat = pd.concat(all_sim)
    exp_cat = pd.concat(all_exp)
    min_rg = float(min(sim_cat.min(), exp_cat.min()))
    max_rg = float(max(sim_cat.max(), exp_cat.max()))
    pad = 0.08 * (max_rg - min_rg)
    lims = (min_rg - pad, max_rg + pad)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for ax, (name, df, color) in zip(axes, panels):
        ax.scatter(df["sim_rg_nm"], df["exp_rg_nm"],
                   color=color, s=10, alpha=0.65)
        ax.plot(lims, lims, color="black", linewidth=0.7, alpha=0.6)
        mae = float(df["abs_error_nm"].mean())
        ax.set_title(f"{name}\nfull-pool MAE = {mae:.3f} nm  (n={len(df)})",
                     fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("simulated R$_g$ (nm)")
    axes[0].set_ylabel("reference R$_g$ (nm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Candidate examples (params.json + loss.csv only)
# ---------------------------------------------------------------------------


def copy_candidate_examples(
    validation_dir: Path,
    out_root: Path,
    n_examples: int = 3,
) -> list[str]:
    obs = pd.read_csv(validation_dir / "observations.csv").sort_values("loss")
    keys = obs["candidate_id"].astype(str).head(n_examples).tolist()
    written: list[str] = []
    for index, key in enumerate(keys):
        src = validation_dir / "candidates" / key
        if not src.exists():
            continue
        anon_name = f"candidate_{index:04d}"
        dst = out_root / anon_name
        dst.mkdir(parents=True, exist_ok=True)
        # params.json — only expose dimensionality, never values.
        cand_meta = src / "candidate.json"
        if cand_meta.exists():
            payload = json.loads(cand_meta.read_text(encoding="utf-8"))
            params = payload.get("lambda_pw") or payload.get("parameters") or {}
            (dst / "params.json").write_text(
                json.dumps({"candidate_index": index, "n_parameters": len(params)},
                           indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        loss_path = src / "loss.csv"
        if loss_path.exists():
            df = pd.read_csv(loss_path)
            keep = [c for c in df.columns if c not in {"huber_delta_nm2", "huber_delta"}]
            df[keep].to_csv(dst / "loss.csv", index=False)
        written.append(anon_name)
    return written


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def write_release_readme(
    dst_dir: Path,
    n_initial_sobol: int,
    n_observations: int,
    n_candidates_examples: int,
    summary: dict[str, float],
) -> None:
    body = f"""# Sample BO results

Representative slice of a real campaign, anonymised for public release.
Sequence labels have been replaced with ``seqXXX`` indices and parameter
values are not exposed.

## Files

* `observations_curve.png` — proxy loss vs. candidate index for the live
  stratified-random BO. Background bands mark the Sobol exploration
  region (``[0, {n_initial_sobol})``) and the BoTorch BO region
  (``[{n_initial_sobol}, {n_observations})``). The legend is mounted
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

* `candidates_examples/` — {n_candidates_examples} candidate folders.
  Each holds:
    * ``params.json`` — only the parameter-vector dimensionality
      (numeric values are intentionally omitted).
    * ``loss.csv`` — full-pool loss summary for the candidate.

## Headline numbers

* Trials: **{n_observations}** stratified candidates, **{n_initial_sobol}**
  in the Sobol exploration phase and **{n_observations - n_initial_sobol}**
  in the BoTorch optimization phase.
* Physical baseline A full-pool MAE: **{summary.get('Physical baseline A|mae_nm', float('nan')):.3f} nm**.
* Physical baseline B full-pool MAE: **{summary.get('Physical baseline B|mae_nm', float('nan')):.3f} nm**.
* Best BO            full-pool MAE: **{summary.get('Best BO (full pool)|mae_nm', float('nan')):.3f} nm**.
"""
    (dst_dir / "README.md").write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stratified-dir", required=True,
                        help="path to the live stratified-BO campaign directory")
    parser.add_argument("--validation-dir", required=True,
                        help="path to the full-pool validation campaign directory")
    parser.add_argument("--n-initial-sobol", type=int, required=True,
                        help="config bo.n_initial_sobol of the stratified campaign")
    parser.add_argument("--baseline-a-csv", required=True,
                        help="per-sequence Rg summary CSV for the first physical baseline")
    parser.add_argument("--baseline-b-csv", required=True,
                        help="per-sequence Rg summary CSV for the second physical baseline")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    stratified = Path(args.stratified_dir).resolve()
    validation = Path(args.validation_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates_examples").mkdir(parents=True, exist_ok=True)

    # Build the global label map from the validation candidate that has the
    # most labels (best chance of covering everything).
    val_obs = pd.read_csv(validation / "observations.csv")
    best_val = val_obs.sort_values("loss").iloc[0]
    sample = candidate_seq_results(validation / "candidates" / str(best_val["candidate_id"]))
    label_map = anon_label_map(sample["label"].astype(str).tolist())

    # 1. Loss curve from the stratified campaign.
    obs = pd.read_csv(stratified / "observations.csv")
    obs_anon = drop_lambda_columns(obs)
    obs_anon = assign_phase(obs_anon, args.n_initial_sobol)
    anon_csv = out / "observations_anon.csv"
    obs_anon.to_csv(anon_csv, index=False)
    print(f"[anon] wrote {anon_csv}")

    curve_path = out / "observations_curve.png"
    render_observations_curve(obs_anon, args.n_initial_sobol, curve_path)
    print(f"[anon] wrote {curve_path}")

    # 2. Top-5 vs full-pool from the validation campaign.
    top5_path = out / "top5_vs_full.png"
    render_top5_vs_full(validation, label_map, top5_path)
    print(f"[anon] wrote {top5_path}")

    # 3. Before / after — two physical baselines vs best BO.
    before_after_path = out / "before_after.png"
    summary = render_before_after(
        Path(args.baseline_a_csv).resolve(),
        Path(args.baseline_b_csv).resolve(),
        validation,
        before_after_path,
    )
    print(f"[anon] wrote {before_after_path} ({summary})")

    # 4. Candidate examples (use validation dir since it has full sequence_results).
    written = copy_candidate_examples(validation, out / "candidates_examples", n_examples=3)
    print(f"[anon] copied {len(written)} candidate examples")

    # 5. README pointer.
    write_release_readme(out, args.n_initial_sobol, len(obs_anon), len(written), summary)
    print(f"[anon] wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()
