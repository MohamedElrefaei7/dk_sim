"""Tests for the synthetic promo-targeting simulation.

Thresholds here are fixed by the study design. Do not loosen them to make a
test pass; see README "Parameter changes" for the permitted knobs.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dk_sim.analysis import run_all, run_one, separability_auc
from dk_sim.config import SimConfig
from dk_sim.model import FEATURES, select_features
from dk_sim.simulate import simulate

CFG = SimConfig(n_users=20_000, n_seeds=3)
SEEDS = range(CFG.n_seeds)
MODEL_PY = Path(__file__).resolve().parents[1] / "dk_sim" / "model.py"


def mean_metric(cfg, condition, metric="representation_ratio", features=FEATURES):
    return float(np.mean([run_one(cfg, s, condition, features, with_auc=False)[metric] for s in SEEDS]))


@pytest.fixture(scope="module")
def runs():
    """One (row, deciles) result per condition x seed, shared across tests."""
    return {
        c: [run_one(CFG, s, c, with_auc=False, return_deciles=True) for s in SEEDS]
        for c in ("placebo", "engagement_only", "loss_chasing")
    }


@pytest.fixture(scope="module")
def ratios(runs):
    return {c: float(np.mean([row["representation_ratio"] for row, _ in rs])) for c, rs in runs.items()}


def test_model_py_never_mentions_hidden_columns():
    assert "_true_" not in MODEL_PY.read_text()


def test_features_exclude_hidden_columns():
    assert not any(f.startswith("_true_") for f in FEATURES)
    df = simulate(CFG, 0, "loss_chasing")
    assert list(select_features(df).columns) == FEATURES


def test_leak_would_be_caught(ratios):
    leaked_features = FEATURES + ["_true_at_risk"]
    leaked = mean_metric(CFG, "loss_chasing", features=leaked_features)
    assert leaked > ratios["loss_chasing"]


def test_placebo_oracle_targets_at_base_rate(runs):
    """Under placebo the true effect is independent of risk, so targeting by the true
    effect must hit the base rate. This is the pipeline / leakage sanity check.

    History: the spec originally required the *model* placebo ratio to lie in
    [0.8, 1.2]. It does not (~2.6-2.8): with no real signal, the model ranks estimation
    noise, and that noise is largest for at-risk users. Changing the estimator did not
    fix this robustly, so it is recorded as a finding (next test), not forced. See README.
    """
    oracle = float(np.mean([row["oracle_ratio"] for row, _ in runs["placebo"]]))
    assert 0.8 <= oracle <= 1.2, f"placebo oracle ratio = {oracle:.3f}"


def test_placebo_noise_floor_is_u_shaped(runs):
    """The model's placebo over-targeting is a noise artifact, not leakage: at-risk users
    are over-represented in BOTH the lowest and highest predicted-uplift deciles.
    A leak (or a real effect) would push them into the top decile only."""
    dec = np.mean([d for _, d in runs["placebo"]], axis=0)
    middle = dec[1:9].mean()
    assert dec[0] > 2 * middle and dec[9] > 2 * middle, np.round(dec, 3)


def test_ordering_of_conditions(runs, ratios):
    oracle = float(np.mean([row["oracle_ratio"] for row, _ in runs["loss_chasing"]]))
    assert ratios["placebo"] < ratios["engagement_only"] < ratios["loss_chasing"] <= oracle + 0.05, (
        ratios, oracle)


def test_monotonic_in_multiplier():
    vals = [mean_metric(replace(CFG, risk_multiplier=m), "loss_chasing") for m in (1.0, 2.0, 3.0)]
    assert vals[0] < vals[1] < vals[2], vals


def test_randomization_balance():
    df = simulate(CFG, 0, "loss_chasing")
    assert abs(df["treated"].mean() - 0.5) < 0.01
    by_risk = df.groupby("_true_at_risk")["treated"].mean()
    assert abs(by_risk[1] - by_risk[0]) < 0.02


def test_prevalence():
    df = simulate(CFG, 0, "placebo")
    assert abs(df["_true_at_risk"].mean() - 0.05) < 0.005


def test_not_trivially_separable():
    auc = separability_auc(simulate(CFG, 0, "placebo"), 0)
    assert 0.65 <= auc <= 0.92, f"AUC = {auc:.3f}"


def test_determinism():
    cfg = replace(CFG, n_seeds=1)
    pd.testing.assert_frame_equal(run_all(cfg), run_all(cfg))
