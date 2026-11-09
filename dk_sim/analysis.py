"""Run every condition x seed, write results, summary, and figures.

Usage: python -m dk_sim.analysis
"""
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import CONDITIONS, SimConfig
from .model import FEATURES, run_uplift_targeting, select_features, top_fraction_mask
from .simulate import simulate

OUT_DIR = Path("outputs")
METRICS = [
    "base_rate",
    "targeted_share",
    "representation_ratio",
    "oracle_share",
    "oracle_ratio",
    "at_risk_uplift_share",
    "separability_auc",
    "excess_over_placebo",
]
LABELS = {"placebo": "Placebo", "engagement_only": "Engagement-only", "loss_chasing": "Loss-chasing"}


# ---------------- metrics (may read hidden columns) ----------------

def separability_auc(df: pd.DataFrame, seed: int, features: Sequence[str] = FEATURES) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, select_features(df, features), df["_true_at_risk"], cv=cv, scoring="roc_auc")
    return float(scores.mean())


def decile_shares(held: pd.DataFrame, uplift: np.ndarray) -> np.ndarray:
    """At-risk share by predicted-uplift decile (index 0 = lowest decile)."""
    ranks = pd.Series(uplift).rank(method="first").to_numpy()
    dec = pd.qcut(ranks, 10, labels=False)
    risk = held["_true_at_risk"].to_numpy()
    return np.array([risk[dec == d].mean() for d in range(10)])


def run_one(cfg: SimConfig, seed: int, condition: str, features: Sequence[str] = FEATURES,
            with_auc: bool = True, return_deciles: bool = False):
    df = simulate(cfg, seed, condition)
    held, uplift, targeted = run_uplift_targeting(df, seed, cfg.target_frac, cfg.holdout_frac, features)

    risk = held["_true_at_risk"].to_numpy()
    tau = held["_true_tau"].to_numpy()
    oracle = top_fraction_mask(tau, cfg.target_frac)

    base = risk.mean()
    share = risk[targeted].mean()
    row = {
        "condition": condition,
        "seed": seed,
        "risk_multiplier": cfg.risk_multiplier if condition == "loss_chasing" else 1.0,
        "base_rate": base,
        "targeted_share": share,
        "representation_ratio": share / base,
        "oracle_share": risk[oracle].mean(),
        "oracle_ratio": risk[oracle].mean() / base,
        "at_risk_uplift_share": tau[targeted & (risk == 1)].sum() / tau[targeted].sum(),
        "separability_auc": separability_auc(df, seed) if with_auc else np.nan,
    }
    if return_deciles:
        return row, decile_shares(held, uplift)
    return row


def run_all(cfg: SimConfig, conditions: Sequence[str] = CONDITIONS, seeds: Sequence[int] | None = None,
            return_deciles: bool = False):
    seeds = range(cfg.n_seeds) if seeds is None else seeds
    rows, dec_rows = [], []
    for cond in conditions:
        for s in seeds:
            row, dec = run_one(cfg, s, cond, return_deciles=True)
            rows.append(row)
            for d, v in enumerate(dec, start=1):
                dec_rows.append({"condition": cond, "seed": s, "decile": d, "at_risk_share": v})
    results = pd.DataFrame(rows)
    results = add_excess_over_placebo(results)
    if return_deciles:
        return results, pd.DataFrame(dec_rows)
    return results


def add_excess_over_placebo(results: pd.DataFrame) -> pd.DataFrame:
    """Paired (same seed => same population and noise) representation ratio minus the
    placebo ratio. The placebo ratio is the noise floor: what the model produces when
    promos affect everyone equally. NaN when placebo was not run."""
    if "placebo" not in set(results["condition"]):
        results["excess_over_placebo"] = np.nan
        return results
    floor = results[results.condition == "placebo"].set_index("seed")["representation_ratio"]
    results["excess_over_placebo"] = results["representation_ratio"] - results["seed"].map(floor)
    return results


def summarize(results: pd.DataFrame) -> dict:
    out = {}
    for cond, g in results.groupby("condition", sort=False):
        out[cond] = {
            m: {
                "mean": float(g[m].mean()),
                "p2.5": float(np.percentile(g[m], 2.5)),
                "p97.5": float(np.percentile(g[m], 97.5)),
            }
            for m in METRICS
        }
    return out


# ---------------- figures ----------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def fig1(summary: dict, prevalence: float, path: Path):
    plt = _style()
    cmap = plt.get_cmap("tab10")
    series = [
        ("base_rate", "Base rate (held-out)", cmap(7), ""),
        ("targeted_share", "Model-targeted top 10%", cmap(0), "//"),
        ("oracle_share", "Oracle top 10% (true effect)", cmap(1), ".."),
    ]
    x = np.arange(len(CONDITIONS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label, color, hatch) in enumerate(series):
        means, lo, hi = [], [], []
        for c in CONDITIONS:
            m, l, h = (summary[c][key][k] for k in ("mean", "p2.5", "p97.5"))
            means.append(m); lo.append(m - l); hi.append(h - m)
        ax.bar(x + (i - 1) * w, means, w, yerr=[lo, hi], capsize=3, label=label,
               color=color, hatch=hatch, edgecolor="black", linewidth=0.6)
    ax.axhline(prevalence, ls="--", color="black", lw=1, label=f"Population prevalence ({prevalence:.0%})")
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Share of group that is at-risk")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("At-risk share among users targeted for promotions")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig2(deciles: pd.DataFrame, prevalence: float, path: Path):
    plt = _style()
    cmap = plt.get_cmap("tab10")
    styles = {
        "placebo": (cmap(7), "^", ":"),
        "engagement_only": (cmap(0), "o", "-"),
        "loss_chasing": (cmap(1), "s", "-."),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, (color, marker, ls) in styles.items():
        g = deciles[deciles.condition == cond].groupby("decile")["at_risk_share"]
        m = g.mean()
        lo = g.quantile(0.025)
        hi = g.quantile(0.975)
        ax.fill_between(m.index, lo, hi, color=color, alpha=0.2, lw=0)
        label = "Placebo (no real effect)" if cond == "placebo" else LABELS[cond]
        ax.plot(m.index, m.values, color=color, marker=marker, ls=ls, label=label)
    ax.axhline(prevalence, ls="--", color="black", lw=1, label=f"Population prevalence ({prevalence:.0%})")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Predicted-uplift decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Share of decile that is at-risk")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("At-risk share by predicted promo uplift")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(cfg: SimConfig = SimConfig(), out_dir: Path = OUT_DIR):
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    results, deciles = run_all(cfg, return_deciles=True)
    summary = summarize(results)
    results.to_csv(out_dir / "results.csv", index=False)
    deciles.to_csv(out_dir / "deciles.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    fig1(summary, cfg.at_risk_prevalence, out_dir / "fig1_at_risk_share.png")
    fig2(deciles, cfg.at_risk_prevalence, out_dir / "fig2_decile_curve.png")

    table = pd.DataFrame({
        c: {m: f"{v['mean']:.3f} [{v['p2.5']:.3f}, {v['p97.5']:.3f}]" for m, v in summary[c].items()}
        for c in summary
    })
    print(f"Synthetic simulation: n_users={cfg.n_users}, seeds={cfg.n_seeds}, "
          f"target_frac={cfg.target_frac}  (mean [2.5%, 97.5%] across seeds)\n")
    print(table.to_string())
    print(f"\nWall-clock: {time.perf_counter() - t0:.1f}s  -> outputs written to {out_dir}/")
    return results, summary


if __name__ == "__main__":
    main()
